#!/usr/bin/env python3
"""Live silicon verification of Milestone 19C (AgentLoop + desk_task + Tier-0/Tier-1 cache hits) on roma."""

import argparse
import asyncio
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.lfm3_harness import (  # noqa: E402
    DaemonClient,
    HarnessStore,
    MissionRunner,
    WorkspaceSandbox,
    setup_desk_task_workspace,
)


async def run_live_desk_task(args: argparse.Namespace) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"m19c_{args.family}_") as tmp:
        root = Path(tmp)
        ws = root / "ws"
        spec = setup_desk_task_workspace(ws, max_attempts=3, max_rounds_per_attempt=14)
        sandbox = WorkspaceSandbox(ws)
        store = HarnessStore(root / "harness.db")
        ws_row = store.register_workspace(f"live_{args.family}", ws)

        def _log_round(r, raw_text: str) -> None:
            snippet = raw_text.replace("\n", "\\n")[:120]
            print(
                f"  [live] round {r.round_num:2d} [{r.phase:6s}] prompt={r.prompt_tokens:4d} "
                f"cached={r.cached_tokens:4d} ({r.cache_status}/{r.cache_tier} "
                f"{r.cache_restore_ms:.2f}ms) prefill={r.prefill_ms:.1f}ms "
                f"decode={r.decode_ms:.1f}ms calls={r.tool_calls} out={snippet!r}",
                flush=True,
            )

        client = DaemonClient(args.host, args.port, connect_timeout_s=10.0)
        runner = MissionRunner(
            client,
            sandbox,
            store,
            journal_dir=root / "journals",
            family=args.family,  # type: ignore[arg-type]
            on_round_complete=_log_round,
        )

        print(f"=== Running Live desk_task on {args.host}:{args.port} (family={args.family}) ===", flush=True)
        mission_res = await runner.run(ws_row["id"], spec)

        last_job = mission_res.job_results[-1] if mission_res.job_results else None
        round_rows = [asdict(r) for r in (last_job.rounds if last_job else [])]
        later_hits = [r for r in round_rows if r["round_num"] >= 2 and r["cached_tokens"] > 0]

        print(
            f"[mission] status={mission_res.status} attempts={mission_res.attempts} "
            f"spec_sha256={mission_res.spec_sha256[:12]}",
            flush=True,
        )
        for r in round_rows:
            print(
                f"  round {r['round_num']:2d} [{r['phase']:6s}]: prompt={r['prompt_tokens']:4d} "
                f"cached={r['cached_tokens']:4d} ({r['cache_status']} / {r['cache_tier']} "
                f"restore={r['cache_restore_ms']:.2f}ms) prefill={r['prefill_ms']:.1f}ms "
                f"decode={r['decode_ms']:.1f}ms calls={r['tool_calls']}",
                flush=True,
            )
        if last_job and last_job.deliverables:
            for d in last_job.deliverables:
                print(
                    f"  [deliverable] {d.rel_path} bytes={d.bytes} sha256={d.sha256[:12]} read={d.verified_read}",
                    flush=True,
                )
            print(
                f"  [grounding] cited={last_job.numeric_grounding.cited_numbers} "
                f"ratio={last_job.numeric_grounding.grounded_ratio}",
                flush=True,
            )

        out: Dict[str, Any] = {
            "host": args.host,
            "port": args.port,
            "family": args.family,
            "status": mission_res.status,
            "attempts": mission_res.attempts,
            "spec_sha256": mission_res.spec_sha256,
            "checks": [asdict(c) for c in (mission_res.reports[-1].results if mission_res.reports else [])],
            "rounds": round_rows,
            "deliverables": [asdict(d) for d in (last_job.deliverables if last_job else [])],
            "numeric_grounding": asdict(last_job.numeric_grounding) if last_job else None,
            "deliverable_content": (ws / "reports" / "desk_summary.md").read_text(encoding="utf-8")
            if (ws / "reports" / "desk_summary.md").exists()
            else "",
            "error": last_job.error if last_job else "",
        }
        store.close()

        if args.out_json:
            op = Path(args.out_json)
            op.parent.mkdir(parents=True, exist_ok=True)
            op.write_text(json.dumps(out, indent=2), encoding="utf-8")

        assert mission_res.status == "passed", f"desk_task failed on {args.family}: {out}"
        assert len(later_hits) >= 1, f"Expected >= 1 cached prefix hit in rounds 2..N on {args.family}: {round_rows}"
        assert all(r["cache_restore_ms"] < 25.0 for r in later_hits), f"Expected restore < 25 ms: {later_hits}"
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Live roma desk_task verification for Milestone 19C")
    parser.add_argument("--host", default="roma")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--family", choices=["lfm3", "lfm2"], default="lfm3")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()
    asyncio.run(run_live_desk_task(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

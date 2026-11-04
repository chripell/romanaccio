#!/usr/bin/env python3
"""Builds the mixed-precision package FINDINGS 39.3 describes but never shipped.

39.3 claims that keeping `proj2048` (`conv_out`/`attn_out`) in FP16 while the rest stays
INT8/INT16 raises layer-00 cosine from 0.97629 to 0.99876 and restores 4/4 greedy tokens.
The deployed `eswin_pkg_int8` has no FP16 path at all: `build_lfm3_int8_pkg.py` builds every one
of the 8 archetypes from INT8 templates, and the shipped `proj2048` Class B is 84,520 B against
the 147,352 B FP16 archetype. The engine reproduces 39.3's *failing* all-INT8 number exactly
(logits cosine 0.88736760 vs the documented 0.8874), which is how this was found.

Rather than edit the builder in place, this derives a new package from the deployed INT8 one and
changes exactly one variable: proj2048's archetype loadables, its weight regions, and its
boundary dtypes all come from the FP16 package. Everything else is byte-identical to the INT8
package, so a difference in accuracy is attributable to proj2048 precision alone.

Usage:
    python build_mixed_proj2048_fp16.py [--out DIR]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

MODEL_DIR = Path("/home/chri/aixigua/deep/models/LFM3-1.7B")
PKG_INT8 = MODEL_DIR / "eswin_pkg_int8"
PKG_FP16 = MODEL_DIR / "eswin_pkg"
ALIGN = 4096

# proj2048 covers both names it is built from: conv layers' out_proj and attention layers' o_proj.
PROJ_OPS = ("conv_out", "attn_out")


def pad_to(data: bytes, align: int = ALIGN) -> bytes:
    rem = len(data) % align
    return data if rem == 0 else data + b"\x00" * (align - rem)


def read_region(blob: Path, entry: dict) -> bytes:
    off = int(entry["weight_offset"])
    size = int(entry["weight_size"])
    with blob.open("rb") as f:
        f.seek(off)
        got = f.read(size)
    if len(got) != size:
        sys.exit(f"short read from {blob.name} at {off}: {len(got)}/{size}")
    return got


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=MODEL_DIR / "eswin_pkg_mixed")
    args = p.parse_args()

    for d in (PKG_INT8, PKG_FP16):
        if not (d / "weights_map.json").is_file():
            sys.exit(f"missing {d}/weights_map.json -- build the FP16 and INT8 packages first")

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(PKG_INT8, args.out)
    print(f"[+] Copied INT8 package to {args.out}")

    int8_map = json.loads((PKG_INT8 / "weights_map.json").read_text())
    fp16_map = json.loads((PKG_FP16 / "weights_map.json").read_text())
    int8_qs = json.loads((PKG_INT8 / "quant_scales.json").read_text())
    fp16_qs = json.loads((PKG_FP16 / "quant_scales.json").read_text())

    # 1. Archetype loadables: FP16 proj2048 replaces the INT8 ones.
    for t in ("t1", "t64"):
        src = PKG_FP16 / "models" / f"lfm3_proj2048_{t}_classb.model"
        dst = args.out / "models" / f"lfm3_proj2048_{t}_classb.model"
        old = dst.stat().st_size if dst.exists() else 0
        shutil.copy2(src, dst)
        print(f"[+] proj2048_{t}_classb.model: {old} -> {dst.stat().st_size} B (FP16)")

    # 2. Weight regions: rebuild both die blobs in the INT8 map's order, substituting FP16 bytes
    #    for proj2048 entries and re-laying out every offset.
    # The INT8 weights_map carries only {layer, die, weight_offset, weight_size} -- no "op"
    # field -- so proj2048 regions are identified by key suffix. (First version of this script
    # filtered on a missing "op" field, matched nothing, and silently produced an all-INT8
    # package with FP16 loadables. The "0 proj2048 regions" line in its output is what caught it.)
    proj_keys = {k for k in int8_map if k.endswith(PROJ_OPS)}
    if len(proj_keys) != 24:
        sys.exit(f"expected 24 proj2048 regions (13 conv_out + 11 attn_out), found {len(proj_keys)}")
    missing = proj_keys - set(fp16_map)
    if missing:
        sys.exit(f"FP16 package has no region for: {sorted(missing)[:5]}")
    print(f"[+] {len(proj_keys)} proj2048 regions will come from the FP16 package "
          f"({len(int8_map)} regions total)")

    new_map = {}
    for die in (0, 1):
        entries = [(k, v) for k, v in int8_map.items() if int(v["die"]) == die]
        entries.sort(key=lambda kv: int(kv[1]["weight_offset"]))
        out_path = args.out / f"weights_die{die}.bin"
        offset = 0
        with out_path.open("wb") as out:
            for key, ent in entries:
                if key in proj_keys:
                    src_die = PKG_FP16 / f"weights_die{int(fp16_map[key]['die'])}.bin"
                    blob = pad_to(read_region(src_die, fp16_map[key]))
                else:
                    blob = pad_to(read_region(PKG_INT8 / f"weights_die{die}.bin", ent))
                out.write(blob)
                new_ent = dict(ent)
                new_ent["weight_offset"] = offset
                new_ent["weight_size"] = len(blob)
                new_ent["precision"] = "fp16" if key in proj_keys else "int8"
                new_map[key] = new_ent
                offset += len(blob)
        print(f"[+] weights_die{die}.bin: {out_path.stat().st_size:,} B "
              f"(was {(PKG_INT8 / f'weights_die{die}.bin').stat().st_size:,} B)")

    (args.out / "weights_map.json").write_text(json.dumps(new_map, indent=2))

    # 3. Boundary dtypes for proj2048 must match the FP16 loadable, or the engine will pack INT8
    #    activations into an FP16 graph.
    # There is only one quant_scales.json in the tree (INT8-calibrated, stored under eswin_pkg),
    # so the FP16 dtypes cannot be copied from anywhere -- they have to be written. An FP16
    # boundary has no quantization step, so 1.0 is the identity the engine's pack/unpack expects.
    swapped = 0
    for key in sorted(proj_keys):
        ent = int8_qs["subgraphs"].get(key)
        if ent is None:
            sys.exit(f"quant_scales.json has no subgraph entry for {key}")
        ent["input_dtype"] = "fp16"
        ent["output_dtype"] = "fp16"
        ent["input_step"] = 1.0
        ent["output_step"] = 1.0
        swapped += 1
    arch = int8_qs.setdefault("archetype_shared", {}).setdefault("proj2048", {})
    arch.update({"input_dtype": "fp16", "output_dtype": "fp16",
                 "input_step": 1.0, "output_step": 1.0})
    (args.out / "quant_scales.json").write_text(json.dumps(int8_qs, indent=2))
    print(f"[+] Swapped {swapped}/{len(proj_keys)} proj2048 subgraph scale entries to FP16")

    manifest = json.loads((args.out / "manifest.json").read_text())
    manifest["precision"] = "int8"
    manifest["precision_note"] = ("mixed: conv_in/attn_qkv/mlp/lm_head INT8+INT16, "
                                  "proj2048 (conv_out, attn_out) FP16 -- FINDINGS 39.3")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(int(v["weight_size"]) for v in new_map.values())
    print(f"[+] Wrote {args.out} ({total:,} B of weight regions across both dies)")
    print("    Run twice and compare: a deterministic feed-forward engine must not vary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

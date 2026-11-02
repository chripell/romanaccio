#!/usr/bin/env python3
"""Counts distinct Class B "stream classes" in a directory of compiled loadables.

Why this exists: a compiled loadable is not interchangeable between experts, because EsAAC bakes
weight-derived per-op requantization bytes into `kmd-task-op_list` (FINDINGS §33.2). Two experts
can still share one loadable when their whole Class B files are byte-identical -- identical bytes
cannot compute differently, so no board test is needed. The number of such classes, against the
measured ceiling of 62 resident models per die (§33.3), is what decides whether INT8 MoE dispatch
is feasible at all (§34.5).

Usage:
    python count_stream_classes.py DIR [--tag-regex 'L(?P<layer>\\d+)E(?P<expert>\\d+)']

DIR holds `*_classb.model` files (output of `compiler.py --convert-weights`). Filenames are parsed
for layer/expert tags so the report can say whether a class spans layers, which is the question
that decides if 26 layers can share a handful of loadables.
"""

import argparse
import collections
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eswin_compiler"))

from backend.flatbuffer_emitter import ModelReader  # noqa: E402

OP_LIST = "kmd-task-op_list"


def stream_of(path: Path, blob_name: str = OP_LIST) -> bytes:
    model = ModelReader.read_file(str(path))
    for blob in model.blobs:
        if blob.name == blob_name:
            return bytes(blob.data)
    raise SystemExit(f"{path.name}: no {blob_name} blob -- is this a compiled NPU loadable?")


def differing_offsets(a: bytes, b: bytes) -> list:
    return [i for i in range(min(len(a), len(b))) if a[i] != b[i]]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("directory", type=Path)
    p.add_argument("--tag-regex", default=r"L(?P<layer>\d+)E(?P<expert>\d+)",
                   help="named groups 'layer'/'expert' parsed out of each filename")
    p.add_argument("--blob", default=OP_LIST, help="command stream to fingerprint")
    args = p.parse_args()

    files = sorted(args.directory.glob("*_classb.model"))
    if not files:
        print(f"no *_classb.model files in {args.directory}")
        print("run: compiler.py -m <static.model> -o <dir> --convert-weights")
        return 1

    tag = re.compile(args.tag_regex)
    classes = collections.OrderedDict()   # md5 -> list of (path, layer, expert)
    for f in files:
        digest = hashlib.md5(f.read_bytes()).hexdigest()
        m = tag.search(f.name)
        layer = int(m.group("layer")) if m and "layer" in m.groupdict() else None
        expert = int(m.group("expert")) if m and "expert" in m.groupdict() else None
        classes.setdefault(digest, []).append((f, layer, expert))

    print(f"{len(files)} loadables -> {len(classes)} distinct stream class(es)\n")
    reps = []
    for i, (digest, members) in enumerate(
            sorted(classes.items(), key=lambda kv: (-len(kv[1]), kv[0]))):
        layers = sorted({l for _, l, _ in members if l is not None})
        names = ", ".join(f"L{l}E{e}" if l is not None else path.stem
                          for path, l, e in members)
        span = "spans layers " + ",".join(str(x) for x in layers) if len(layers) > 1 else (
            f"layer {layers[0]}" if layers else "layer unknown")
        print(f"  class {i} [{digest[:8]}] {len(members):>2}/{len(files)} members, {span}")
        print(f"      {names}")
        reps.append((digest, members[0][0]))

    # How far apart are the classes? If they differ in only a few stream bytes, a future loader
    # could in principle hold one file per class cheaply; if they differ widely, something else
    # than the requant bytes is expert-specific and the model of the problem is wrong.
    if len(reps) > 1:
        print(f"\npairwise differences between class representatives ({args.blob}):")
        streams = {d: stream_of(p, args.blob) for d, p in reps}
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                d1, p1 = reps[i]
                d2, p2 = reps[j]
                s1, s2 = streams[d1], streams[d2]
                if len(s1) != len(s2):
                    print(f"  {d1[:8]} vs {d2[:8]}: stream sizes differ "
                          f"({len(s1)} vs {len(s2)})")
                    continue
                offs = differing_offsets(s1, s2)
                vals1 = sorted({s1[o] for o in offs})
                vals2 = sorted({s2[o] for o in offs})
                print(f"  {d1[:8]} vs {d2[:8]}: {len(offs):>3} bytes differ; "
                      f"values {vals1} -> {vals2}; offsets {offs[:8]}"
                      f"{' ...' if len(offs) > 8 else ''}")
                whole1, whole2 = p1.read_bytes(), p2.read_bytes()
                if len(whole1) == len(whole2):
                    n = sum(1 for x, y in zip(whole1, whole2) if x != y)
                    print(f"{'':>22}whole file: {n} of {len(whole1):,} bytes differ")

    layers_seen = sorted({l for ms in classes.values() for _, l, _ in ms if l is not None})
    if layers_seen:
        print(f"\ncoverage: layers {layers_seen}; "
              f"{len(classes)} class(es) over {len(layers_seen)} layer(s) "
              f"= {len(classes) / len(layers_seen):.2f} classes/layer")
        print("Residency budget: 62 models per die (FINDINGS §33.3). A 26-MoE-layer model needs "
              "every layer's classes resident,")
        print("so the extrapolated total -- not the per-layer figure -- is what must stay under 62.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

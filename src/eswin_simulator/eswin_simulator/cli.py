"""
Command-Line Interface for ESWIN Simulator (eswin-sim).
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure package root is in sys.path
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from eswin_simulator.loader.loadable_reader import LoadableModel
from eswin_simulator.memory.address_space import VirtualAddressSpace


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.2f} KiB"
    elif num_bytes < 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MiB"
    else:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GiB"


def print_banner() -> None:
    print("=" * 80)
    print("  ESWIN EIC7700X Open NPU/DSP Functional Simulator (eswin-sim)")
    print("  Milestone 1: Model Loader & Virtual Address Space Inspector")
    print("=" * 80)


def inspect_model(model_path: str, as_json: bool = False, verbose: bool = False) -> int:
    """Inspects and prints comprehensive metadata for a .model file."""
    p = Path(model_path)
    if not p.is_file():
        print(f"[-] Error: Model file not found: {model_path}", file=sys.stderr)
        return 1

    try:
        model = LoadableModel.from_file(p)
    except Exception as e:
        print(f"[-] Error loading model file {model_path}: {e}", file=sys.stderr)
        return 1

    vas = VirtualAddressSpace(model)
    stats = vas.stats()

    if as_json:
        data = model.summary()
        data["memory_stats"] = stats
        if verbose:
            data["memory_regions"] = [
                {
                    "id": r.id,
                    "domain": r.domain,
                    "size": r.size,
                    "flags": r.flags,
                    "alignment": r.alignment,
                    "contents": r.contents,
                }
                for r in model.memory_regions.values()
            ]
            data["blobs"] = [
                {"name": b.name, "size": b.size, "interface": b.interface}
                for b in model.blobs.values()
            ]
        print(json.dumps(data, indent=2))
        return 0

    print_banner()
    print(f"\n[Model Information]")
    print(f"  • File Path         : {p.resolve()}")
    print(f"  • Container Magic   : NVDA [OK]")
    print(f"  • Loadable Version  : {model.version[0]}.{model.version[1]}.{model.version[2]}")
    print(f"  • File Size         : {format_bytes(os.path.getsize(p))}")
    print(f"  • Tasks / Submodels : {len(model.tasks)}")
    print(f"  • Blobs Embedded    : {len(model.blobs)}")

    # Input Tensors
    print(f"\n[Input Tensors ({len(model.input_tensors)})]")
    if model.input_tensors:
        print(f"  {'Name':<24} {'Shape':<18} {'Type':<8} {'Format':<8} {'Size':<12} {'Domain':<8}")
        print("  " + "-" * 78)
        for t in model.input_tensors:
            shape_str = f"({t.n}, {t.c}, {t.h}, {t.w})"
            print(f"  {t.name:<24} {shape_str:<18} {t.data_type:<8} {t.data_format:<8} {format_bytes(t.size):<12} {t.memory_domain:<8}")
    else:
        print("  (None)")

    # Output Tensors
    print(f"\n[Output Tensors ({len(model.output_tensors)})]")
    if model.output_tensors:
        print(f"  {'Name':<24} {'Shape':<18} {'Type':<8} {'Format':<8} {'Size':<12} {'Domain':<8}")
        print("  " + "-" * 78)
        for t in model.output_tensors:
            shape_str = f"({t.n}, {t.c}, {t.h}, {t.w})"
            print(f"  {t.name:<24} {shape_str:<18} {t.data_type:<8} {t.data_format:<8} {format_bytes(t.size):<12} {t.memory_domain:<8}")
    else:
        print("  (None)")

    # Simulated Memory Layout
    print(f"\n[Simulated Memory Utilization]")
    print(f"  • DRAM Regions (SYSTEM) : {stats['dram_allocated_mb']} MiB ({stats['dram_allocated_bytes']} bytes across {len([r for r in model.memory_regions.values() if r.domain_id == 0])} regions)")
    print(f"  • SPRAM (On-Chip SRAM)  : {format_bytes(stats['spram_used_bytes'])} / 4.00 MiB ({stats['spram_used_percent']}%)")
    print(f"  • Total Address IDs     : {stats['addresses_count']}")
    print(f"  • Relocation Entries    : {len(model.relocations)}")

    # Tasks / Submodels
    print(f"\n[Tasks & Submodels ({len(model.tasks)})]")
    print(f"  {'Task ID':<10} {'Interface':<12} {'KMD ID':<10} {'Address Count':<14}")
    print("  " + "-" * 48)
    for t in model.tasks:
        print(f"  {t.id:<10} {t.interface:<12} {t.kmd_id:<10} {len(t.address_list):<14}")

    # Blobs Overview
    print(f"\n[Embedded Blobs Catalog ({len(model.blobs)})]")
    print(f"  {'Blob Name':<32} {'Size':<14} {'Interface':<10}")
    print("  " + "-" * 58)
    for b in list(model.blobs.values())[:15]:
        print(f"  {b.name:<32} {format_bytes(b.size):<14} {b.interface:<10}")
    if len(model.blobs) > 15:
        print(f"  ... and {len(model.blobs) - 15} more blobs.")

    if verbose:
        print(f"\n[Detailed Memory Regions Table]")
        print(f"  {'ID':<5} {'Domain':<8} {'Size':<12} {'Align':<8} {'Flags':<8} {'Contents'}")
        print("  " + "-" * 70)
        for r in model.memory_regions.values():
            contents_str = ", ".join(r.contents) if r.contents else "-"
            print(f"  {r.id:<5} {r.domain:<8} {format_bytes(r.size):<12} {r.alignment:<8} {r.flags:<8} {contents_str}")

        print(f"\n[Relocation Table Summary ({len(model.relocations)} entries)]")
        print(f"  {'Addr ID':<10} {'Write ID':<10} {'Offset':<12} {'Type':<8}")
        print("  " + "-" * 42)
        for rel in model.relocations[:10]:
            print(f"  {rel.address_id:<10} {rel.write_id:<10} {rel.offset:<12} {rel.reloc_type:<8}")
        if len(model.relocations) > 10:
            print(f"  ... and {len(model.relocations) - 10} more relocations.")

    print("\n" + "=" * 80)
    print("  Status: Model structure validated and parsed successfully!")
    print("=" * 80 + "\n")
    return 0


def main(args: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ESWIN EIC7700X Open NPU/DSP Functional Simulator (eswin-sim)"
    )
    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to ESWIN .model loadable container."
    )
    parser.add_argument(
        "-i", "--info",
        action="store_true",
        help="Inspect model metadata, tensors, memory layout, and blobs."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output model inspection as JSON."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Display detailed memory regions and relocation tables."
    )

    parsed = parser.parse_args(args)

    if parsed.info or parsed.json or True:
        # Default to inspection mode for Milestone 1
        return inspect_model(parsed.model, as_json=parsed.json, verbose=parsed.verbose)

    return 0


if __name__ == "__main__":
    sys.exit(main())

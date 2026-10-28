#!/usr/bin/env python3
"""Synthesize an EsAAC `--quant-stats` table for INT8 compilation, without EsQuant.

The vendor flow is EsQuant -> table.json -> EsAAC --quant-stats. EsQuant's wheel here is
image-dataset oriented (cv2 / DatasetImageList / ImagenetOpencv) and needs its own
container, but the table format is simple enough to produce directly. Reverse-engineered
from the release sample `EIC7x_Release_20250730_nn/sample/yolov3/esaac/table.json`:

  version      "1.0"
  model_info   {inputs, input_data_type, outputs, output_data_type}
  node_info    per graph node: {data_type, input_blob_names, output_blob_names,
                                optimization_option, bias_mode, quantized_method, psum_trunc}
  blob_info    per tensor: {int8_step, int16_step, max, threshold}

Checked across all 247 blob entries of that sample: **int8_step == threshold / 127 in
every case** (0 violations), i.e. symmetric absmax quantization with no zero-point. So a
threshold per tensor is sufficient to define the INT8 scale.

Thresholds come from two sources:
  * weights / initializers — absmax of the actual data, no calibration needed.
  * activations — absmax observed by running the model under onnxruntime over calibration
    inputs, with every intermediate tensor exposed as a graph output.

Calibration inputs are random normal, which is a *proxy*: real hidden states entering an
expert FFN are post-RMSNorm and roughly unit-scale, so the distribution is plausible, but
these thresholds are not tuned on real DeepSeek-V2-Lite activations. Treat the resulting
model as structurally correct and numerically provisional until calibrated on real hidden
states.

Usage:
    python make_quant_table.py expert_ffn.onnx -o table.json [--samples 8] [--seed 0]

    # one loadable serving every expert of a layer: reuse one activation calibration
    python make_quant_table.py expert1.onnx -o expert1_shared.json \
        --activations-from expert0_table.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import shape_inference

# Values mirrored from the vendor sample's node_info entries.
NODE_TEMPLATE = {
    "data_type": "int8",
    "optimization_option": "option2",
    "bias_mode": 1,
    "quantized_method": "per_channel",
    "psum_trunc": 0,
}


def expose_intermediates(model: onnx.ModelProto) -> onnx.ModelProto:
    """Returns a copy whose graph also outputs every intermediate tensor.

    onnxruntime only reports declared outputs, so calibration needs the intermediates
    added explicitly. Tensors that are already outputs, or that are initializers, are
    skipped.
    """
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    declared = {o.name for o in g.output}
    initializer_names = {i.name for i in g.initializer}
    produced = set()
    for node in g.node:
        for o in node.output:
            if o:
                produced.add(o)

    for name in sorted(produced - declared - initializer_names):
        g.output.extend([onnx.helper.make_empty_tensor_value_info(name)])
    return m


def calibrate(model: onnx.ModelProto, input_name: str, shape, samples: int, seed: int):
    """Runs the model over random inputs and returns absmax per named tensor."""
    m = expose_intermediates(model)
    m = shape_inference.infer_shapes(m)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), sess_options=so,
                                providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]

    rng = np.random.default_rng(seed)
    peaks = {n: 0.0 for n in out_names}
    input_peak = 0.0
    for _ in range(samples):
        # Unit-normal, matching post-RMSNorm hidden states entering an expert FFN.
        x = rng.standard_normal(shape).astype(np.float32)
        input_peak = max(input_peak, float(np.max(np.abs(x.astype(np.float64)))))
        for name, value in zip(out_names, sess.run(out_names, {input_name: x})):
            v = float(np.max(np.abs(np.asarray(value, dtype=np.float64))))
            if v > peaks[name]:
                peaks[name] = v
    # A graph input is not produced by any node, so expose_intermediates() never adds it and
    # onnxruntime never reports it. Without this line the input silently fell back to a
    # threshold of 1.0, which clips real post-RMSNorm hidden states (absmax ~3-5).
    peaks[input_name] = max(peaks.get(input_name, 0.0), input_peak)
    return peaks


def build_table(model: onnx.ModelProto, peaks, dtype: str = "int8",
                activations_from: dict | None = None,
                merge_max: dict | None = None,
                quant_method: str = "per_channel") -> dict:
    g = model.graph
    input_names = [i.name for i in g.input]
    output_names = [o.name for o in g.output]
    initializer_names = {i.name for i in g.initializer}

    blob_info = {}

    def add_blob(name: str, threshold: float) -> None:
        threshold = float(max(threshold, 1e-8))   # a zero range would divide by zero
        blob_info[name] = {
            "int8_step": threshold / 127.0,
            "int16_step": threshold / 32767.0,
            "max": threshold,
            "threshold": threshold,
        }

    # Weights: absmax straight from the initializer data.
    for init in g.initializer:
        arr = onnx.numpy_helper.to_array(init)
        add_blob(init.name, float(np.max(np.abs(arr.astype(np.float64)))))

    # Activations, inputs and outputs: from the calibration pass.
    for name in input_names + output_names:
        if name in peaks:
            add_blob(name, peaks[name])
        elif name not in blob_info:
            add_blob(name, 1.0)
    for name, peak in peaks.items():
        if name not in blob_info:
            add_blob(name, peak)

    # Shared-activation mode: one loadable has to serve every expert of a layer, and the
    # activation thresholds are baked into its command streams, so they cannot be per-expert.
    # Take them from a reference table instead; weight thresholds stay this expert's own,
    # because the weight region carries its own scales and is bound at submit time.
    if activations_from is not None:
        ref = activations_from["blob_info"]
        shared = []
        for name in list(blob_info):
            if name in initializer_names:
                continue
            if name not in ref:
                raise SystemExit(f"reference table has no activation blob '{name}' -- it was "
                                 f"calibrated from a different graph")
            blob_info[name] = dict(ref[name])
            shared.append(name)
        print(f"  shared activation thresholds from reference: {', '.join(sorted(shared))}")

    # Merge-max mode: widen every threshold this blob shares with the reference to the larger of
    # the two. Run it once per expert against the others and all experts end up with an
    # IDENTICAL table, which is what lets one compiled loadable serve all of them: measured on
    # real V2-Lite experts, binding expert 1's region into expert 0's loadable does NOT reproduce
    # expert 1's own result unless the weight thresholds match too (FINDINGS §33). Taking the max
    # costs a little precision for the expert with the smaller absmax and buys interchangeability.
    if merge_max is not None:
        ref = merge_max["blob_info"]
        widened = []
        for name, entry in blob_info.items():
            r = ref.get(name)
            if r is None:
                continue
            if r["threshold"] > entry["threshold"]:
                blob_info[name] = dict(r)
                widened.append(name)
        print(f"  widened to the reference maximum: "
              f"{', '.join(sorted(widened)) if widened else '(none)'}")

    node_info = {}
    for node in g.node:
        entry = dict(NODE_TEMPLATE)
        entry["quantized_method"] = quant_method
        entry["input_blob_names"] = [i for i in node.input if i]
        entry["output_blob_names"] = [o for o in node.output if o]
        node_info[node.name or f"{node.op_type}_{len(node_info)}"] = entry

    return {
        "version": "1.0",
        "model_info": {
            "inputs": input_names,
            "input_data_type": [dtype] * len(input_names),
            "outputs": output_names,
            "output_data_type": [dtype] * len(output_names),
        },
        "node_info": node_info,
        "blob_info": blob_info,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx_model", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("table.json"))
    p.add_argument("--samples", type=int, default=8, help="calibration runs (default 8)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--activations-from", type=Path, default=None,
                   help="Take activation thresholds from this existing table instead of this "
                        "model's own calibration, keeping this model's weight thresholds. Use "
                        "it when one compiled loadable must serve several experts: the "
                        "activation scales are baked into its command streams, while the weight "
                        "scales travel inside the submit-time weight region.")
    p.add_argument("--merge-max", type=Path, action="append", default=None, metavar="TABLE",
                   help="Widen every shared blob's threshold to the maximum of this table's "
                        "and the current one. Repeatable. Give every expert of a layer the same "
                        "set of references and they all end up with an identical table, which is "
                        "what makes one compiled loadable serve all of them.")
    p.add_argument("--quant-method", choices=["per_channel", "per_tensor"],
                   default="per_channel",
                   help="Weight quantization granularity written into every node_info entry "
                        "(default per_channel, mirroring the vendor sample). per_channel derives "
                        "scales from the actual weight data, which leaves a weight-dependent "
                        "requant shift in the compiled command stream and so pins a loadable to "
                        "one expert; per_tensor is what lets a whole layer share one loadable.")
    args = p.parse_args()

    model = onnx.load(str(args.onnx_model))
    g = model.graph
    if len(g.input) != 1:
        raise SystemExit(f"expected exactly one graph input, got {len(g.input)}")
    in_name = g.input[0].name
    shape = tuple(d.dim_value for d in g.input[0].type.tensor_type.shape.dim)
    if any(s <= 0 for s in shape):
        raise SystemExit(f"input shape must be static for EsAAC, got {shape}")

    print(f"calibrating {args.onnx_model.name}: input '{in_name}' shape {shape}, "
          f"{args.samples} samples")
    peaks = calibrate(model, in_name, shape, args.samples, args.seed)

    ref_table = None
    if args.activations_from:
        ref_table = json.loads(args.activations_from.read_text())
        print(f"activation thresholds from {args.activations_from.name}")

    merged = None
    if args.merge_max:
        merged = {"blob_info": {}}
        for path in args.merge_max:
            for name, entry in json.loads(path.read_text())["blob_info"].items():
                cur = merged["blob_info"].get(name)
                if cur is None or entry["threshold"] > cur["threshold"]:
                    merged["blob_info"][name] = entry
        print(f"merge-max references: {', '.join(p.name for p in args.merge_max)}")

    table = build_table(model, peaks, activations_from=ref_table, merge_max=merged,
                        quant_method=args.quant_method)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))

    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"  blobs={len(table['blob_info'])} nodes={len(table['node_info'])} "
          f"inputs={table['model_info']['inputs']} outputs={table['model_info']['outputs']}")
    weights = [i.name for i in g.initializer]
    for name in weights:
        b = table["blob_info"][name]
        print(f"  weight {name:<8} threshold={b['threshold']:.6f} int8_step={b['int8_step']:.3e}")
    for name in table["model_info"]["inputs"] + table["model_info"]["outputs"]:
        b = table["blob_info"].get(name)
        if b:
            print(f"  tensor {name:<8} threshold={b['threshold']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    for _ in range(samples):
        # Unit-normal, matching post-RMSNorm hidden states entering an expert FFN.
        x = rng.standard_normal(shape).astype(np.float32)
        for name, value in zip(out_names, sess.run(out_names, {input_name: x})):
            v = float(np.max(np.abs(np.asarray(value, dtype=np.float64))))
            if v > peaks[name]:
                peaks[name] = v
    return peaks


def build_table(model: onnx.ModelProto, peaks, dtype: str = "int8") -> dict:
    g = model.graph
    input_names = [i.name for i in g.input]
    output_names = [o.name for o in g.output]

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

    node_info = {}
    for node in g.node:
        entry = dict(NODE_TEMPLATE)
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

    table = build_table(model, peaks)
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

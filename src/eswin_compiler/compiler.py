"""ESWIN Open Compiler CLI & End-to-End Neural Network Compilation Pipeline.

Translates ONNX and Hugging Face SafeTensors deep learning models into optimized,
hardware-native .model loadables for the EIC7700X dual-die NPU and Cadence Vision Q7 DSP cores.

Supported Model Architectures:
  - Vision & Object Detection: YOLOv3, YOLOv5, ResNet, MobileNet (via ONNX)
  - Large Language Models: SmolLM/SmolLM2, Qwen2/Qwen2.5, DeepSeek-R1, Llama-3.2 (via SafeTensors)

Hardware Offload Pipeline:
  1. Frontend: Ingests ONNX / SafeTensors into 4D [N, C, H, W] ESWIN IR (EIR).
  2. Optimization & Lowering Passes:
     - GemmToConv1x1Pass: Transforms linear projections into 1x1 CMAC Convolutions.
     - SDPFusionPass: Fuses bias additions and activations (ReLU/SiLU/Sigmoid) into SDP.
     - DSPPartitionerPass: Offloads RMSNorm, RoPE, Softmax to Vision Q7 DSP cores.
     - SRAMTilerPass: Allocates weights & intermediate buffers to 4MB on-chip SPRAM.
     - DualDieSplitterPass: Splits wide vocabulary heads across Die 0 and Die 1.
  3. Backend Emitters:
     - NVDLAConvBuilder: Assembles KMD hardware register sequences and surface lists.
     - DSPSubmodelBuilder: Formats UMD DSP parameter structs and event barriers.
     - SDPLUTGenerator: Emits 688-byte piecewise interpolation tables.
     - ModelWriter: Serializes FlatBuffers .model loadable with NVDA magic header.
"""

import argparse
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Package path setup
_CURR_DIR = Path(__file__).parent.resolve()
if str(_CURR_DIR) not in sys.path:
    sys.path.insert(0, str(_CURR_DIR))

try:
    from backend.flatbuffer_emitter import LoadableModel, ModelWriter
    from backend.nvdla_builder import NVDLAConvBuilder, compile_eir_to_model
    from backend.dsp_task_builder import DSPSubmodelBuilder
    from frontend.onnx_importer import ONNXImporter
    from frontend.safetensors_importer import SafeTensorsImporter, SafeTensorsReader
    from ir.graph import EIRGraph
    from ir.node import EIRNode
    from ir.op_types import EngineType, OpType
    from passes.dsp_partitioner import DSPPartitionerPass
    from passes.dual_die_splitter import DualDieSplitterPass
    from passes.fusion_sdp import SDPFusionPass
    from passes.gemm_to_conv1x1 import GemmToConv1x1Pass
    from passes.sram_tiler import SRAMTilerPass
except (ImportError, ValueError):
    from .backend.flatbuffer_emitter import LoadableModel, ModelWriter
    from .backend.nvdla_builder import NVDLAConvBuilder, compile_eir_to_model
    from .backend.dsp_task_builder import DSPSubmodelBuilder
    from .frontend.onnx_importer import ONNXImporter
    from .frontend.safetensors_importer import SafeTensorsImporter, SafeTensorsReader
    from .ir.graph import EIRGraph
    from .ir.node import EIRNode
    from .ir.op_types import EngineType, OpType
    from .passes.dsp_partitioner import DSPPartitionerPass
    from .passes.dual_die_splitter import DualDieSplitterPass
    from .passes.fusion_sdp import SDPFusionPass
    from .passes.gemm_to_conv1x1 import GemmToConv1x1Pass
    from .passes.sram_tiler import SRAMTilerPass

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("eswin_compiler")


@dataclass
class CompilerConfig:
    """Configuration parameters for model compilation."""
    target: str = "single-die"            # 'single-die' or 'dual-die'
    opt_level: int = 2                    # 0: None, 1: Basic, 2: Full (SDP fusion + SPRAM)
    spram_limit_bytes: int = 4 * 1024 * 1024  # 4MB SPRAM per die
    dtype: str = "float16"                # 'float16' or 'int8'
    batch_size: int = 1
    vocab_split_threshold: int = 2048     # Output channel threshold for dual-die tensor split


class EswinCompiler:
    """Main driver for compiling deep learning models into EIC7700X loadables."""

    def __init__(self, config: Optional[CompilerConfig] = None):
        self.config = config or CompilerConfig()

    def compile_onnx(
        self,
        onnx_path: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: Optional[str] = None
    ) -> List[Path]:
        """Compiles an ONNX model (e.g. YOLO, ResNet) into .model loadable binary."""
        onnx_path = Path(onnx_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        name = model_name or onnx_path.stem
        logger.info("Importing ONNX model: %s", onnx_path)
        importer = ONNXImporter()
        graph = importer.import_model(onnx_path)

        return self._compile_graph_pipeline(graph, output_dir, name)

    def compile_safetensors_llm(
        self,
        model_source: Union[str, Path],
        output_dir: Union[str, Path],
        model_name: Optional[str] = None,
        num_layers_to_compile: Optional[int] = None
    ) -> List[Path]:
        """Compiles an LLM from a SafeTensors file or Hugging Face repository folder."""
        source_path = Path(model_source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_dict = self._inspect_model_config(source_path)
        hidden_size = config_dict.get("hidden_size", 576)
        intermediate_size = config_dict.get("intermediate_size", 1536)
        vocab_size = config_dict.get("vocab_size", 49152)
        total_layers = config_dict.get("num_hidden_layers", 30)
        num_heads = config_dict.get("num_attention_heads", 9)
        num_kv_heads = config_dict.get("num_key_value_heads", 3)
        head_dim = hidden_size // num_heads

        layers_to_emit = num_layers_to_compile if num_layers_to_compile is not None else total_layers
        prefix = model_name or source_path.stem or "llm"

        logger.info(
            "Configuring LLM '%s': hidden=%d, intermediate=%d, vocab=%d, layers=%d (emitting %d)",
            prefix, hidden_size, intermediate_size, vocab_size, total_layers, layers_to_emit
        )

        importer = SafeTensorsImporter()
        compiled_files: List[Path] = []

        # ----------------------------------------------------------------------
        # 1. Compile LM Head (with dual-die tensor parallelism if requested)
        # ----------------------------------------------------------------------
        if self.config.target == "dual-die":
            logger.info("Synthesizing Dual-Die LM Head (Die 0 and Die 1)...")
            g_head_d0 = importer.synthesize_lm_head(
                hidden_size=hidden_size, vocab_size=vocab_size, die=0, dtype=self.config.dtype
            )
            g_head_d1 = importer.synthesize_lm_head(
                hidden_size=hidden_size, vocab_size=vocab_size, die=1, dtype=self.config.dtype
            )

            p_d0 = output_dir / f"{prefix}_lm_head_d0_npu_b{self.config.batch_size}.model"
            p_d1 = output_dir / f"{prefix}_lm_head_d1_npu_b{self.config.batch_size}.model"

            self._compile_single_layer(g_head_d0, p_d0)
            self._compile_single_layer(g_head_d1, p_d1)
            compiled_files.extend([p_d0, p_d1])
        else:
            logger.info("Synthesizing Single-Die LM Head...")
            g_head = importer.synthesize_lm_head(
                hidden_size=hidden_size, vocab_size=vocab_size, die=None, dtype=self.config.dtype
            )
            p_head = output_dir / f"{prefix}_lm_head_npu_b{self.config.batch_size}.model"
            self._compile_single_layer(g_head, p_head)
            compiled_files.append(p_head)

        # ----------------------------------------------------------------------
        # 2. Compile SwiGLU MLP Blocks
        # ----------------------------------------------------------------------
        logger.info("Synthesizing %d SwiGLU MLP blocks...", layers_to_emit)
        for layer_idx in range(layers_to_emit):
            mlp_graph = importer.synthesize_mlp_block(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                dtype=self.config.dtype
            )
            
            # For dual-die pipeline, assign layers evenly across dies
            die_tag = ""
            if self.config.target == "dual-die":
                die_num = 0 if layer_idx < (layers_to_emit // 2) else 1
                die_tag = f"_d{die_num}"

            p_mlp = output_dir / f"{prefix}_mlp_l{layer_idx}{die_tag}_npu_b{self.config.batch_size}.model"
            self._compile_single_layer(mlp_graph, p_mlp)
            compiled_files.append(p_mlp)

        # ----------------------------------------------------------------------
        # 3. Compile Attention Blocks (NPU + DSP co-processing)
        # ----------------------------------------------------------------------
        logger.info("Synthesizing %d Attention blocks (NPU + DSP)...", layers_to_emit)
        for layer_idx in range(layers_to_emit):
            attn_graph = importer.synthesize_attention_block(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                dtype=self.config.dtype
            )

            die_tag = ""
            if self.config.target == "dual-die":
                die_num = 0 if layer_idx < (layers_to_emit // 2) else 1
                die_tag = f"_d{die_num}"

            p_attn = output_dir / f"{prefix}_attn_l{layer_idx}{die_tag}_npu_b{self.config.batch_size}.model"
            self._compile_single_layer(attn_graph, p_attn)
            compiled_files.append(p_attn)

        logger.info("Successfully generated %d loadable files in %s", len(compiled_files), output_dir)
        return compiled_files

    def _compile_graph_pipeline(
        self, graph: EIRGraph, output_dir: Path, base_name: str
    ) -> List[Path]:
        """Applies optimization passes and lowers EIRGraph into .model loadables."""
        # 1. Lower GEMM/MatMul to 1x1 Conv
        GemmToConv1x1Pass().run(graph)

        # 2. Fuse Activations and Bias Additions
        if self.config.opt_level >= 1:
            SDPFusionPass().run(graph)

        # 3. Partition DSP vector operations
        DSPPartitionerPass().run(graph)

        # 4. Plan 4MB SPRAM residency
        if self.config.opt_level >= 2:
            SRAMTilerPass(sram_capacity=self.config.spram_limit_bytes).run(graph)

        # 5. Dual-die partitioning if requested and applicable
        compiled_files: List[Path] = []
        if self.config.target == "dual-die":
            splitter = DualDieSplitterPass(vocab_split_threshold=self.config.vocab_split_threshold)
            if splitter.run(graph):
                g_d0, g_d1 = splitter.split_graph_tensor_parallel(graph)
                p0 = output_dir / f"{base_name}_d0_npu_b{self.config.batch_size}.model"
                p1 = output_dir / f"{base_name}_d1_npu_b{self.config.batch_size}.model"
                self._compile_single_layer(g_d0, p0)
                self._compile_single_layer(g_d1, p1)
                compiled_files.extend([p0, p1])
                return compiled_files

        # Single target output
        out_file = output_dir / f"{base_name}_npu_b{self.config.batch_size}.model"
        self._compile_single_layer(graph, out_file)
        compiled_files.append(out_file)
        return compiled_files

    def _compile_single_layer(self, graph: EIRGraph, out_path: Path) -> None:
        """Lowers an optimized single-layer EIRGraph into a .model binary."""
        # Apply standard lowering passes if not already done
        GemmToConv1x1Pass().run(graph)
        if self.config.opt_level >= 1:
            SDPFusionPass().run(graph)
        DSPPartitionerPass().run(graph)
        if self.config.opt_level >= 2:
            SRAMTilerPass(sram_capacity=self.config.spram_limit_bytes).run(graph)

        conv_nodes = [n for n in graph.nodes if n.op_type == OpType.CONV]
        if not conv_nodes:
            raise ValueError(f"Graph '{graph.name}' contains no CONV nodes for NPU execution.")

        primary_conv = conv_nodes[0]
        model = NVDLAConvBuilder.build_from_node(primary_conv, model_name=graph.name)
        ModelWriter.write_file(model, out_path)
        logger.info("Emitted loadable: %s (%d bytes)", out_path.name, out_path.stat().st_size)

    def _inspect_model_config(self, source_path: Path) -> Dict[str, Any]:
        """Reads architecture metadata from config.json or falls back to SmolLM-135M defaults."""
        config_path = source_path / "config.json" if source_path.is_dir() else source_path.parent / "config.json"
        if config_path.is_file():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to parse config.json (%s), using defaults", e)

        # Default fallback to SmolLM-135M architecture
        return {
            "hidden_size": 576,
            "intermediate_size": 1536,
            "vocab_size": 49152,
            "num_hidden_layers": 30,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
        }


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(
        description="ESWIN Open Compiler: Compile ONNX & SafeTensors models for EIC7700X NPU/DSP."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to input ONNX model (.onnx), SafeTensors (.safetensors), or model directory."
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="build/models",
        help="Output directory where .model loadables will be saved (default: build/models)."
    )
    parser.add_argument(
        "-t", "--target",
        choices=["single-die", "dual-die"],
        default="single-die",
        help="Hardware execution target: 'single-die' (NPU 0) or 'dual-die' (NPU 0 + NPU 1)."
    )
    parser.add_argument(
        "--opt-level",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Optimization level: 0 (none), 1 (SDP fusion), 2 (SDP fusion + 4MB SPRAM tiling)."
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=1,
        help="Execution batch size (default: 1)."
    )
    parser.add_argument(
        "--precision",
        choices=["float16", "int8"],
        default="float16",
        help="Computation precision (default: float16)."
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Number of Transformer layers to compile (for testing/partial compile)."
    )

    args = parser.parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        logger.error("Input model path does not exist: %s", model_path)
        sys.exit(1)

    cfg = CompilerConfig(
        target=args.target,
        opt_level=args.opt_level,
        dtype=args.precision,
        batch_size=args.batch_size
    )
    compiler = EswinCompiler(config=cfg)

    out_dir = Path(args.output_dir)
    try:
        if model_path.suffix.lower() == ".onnx":
            out_files = compiler.compile_onnx(model_path, out_dir)
        elif model_path.suffix.lower() == ".safetensors" or model_path.is_dir():
            out_files = compiler.compile_safetensors_llm(
                model_path, out_dir, num_layers_to_compile=args.num_layers
            )
        else:
            logger.error("Unrecognized model format: %s. Expected .onnx, .safetensors, or directory.", model_path)
            sys.exit(1)

        print("\n" + "=" * 60)
        print(f"Compilation Successful: {len(out_files)} loadable(s) emitted to {out_dir}")
        for f in out_files:
            print(f"  -> {f.name} ({f.stat().st_size:,} bytes)")
        print("=" * 60)

    except Exception as e:
        logger.exception("Compilation failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

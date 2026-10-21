"""
End-to-end multi-layer pipeline and model verification tests (Milestone 6).
Validates full coordination across CMAC, CACC, SDP, PDP, Rubik, and DSP engines
on SmolLM-135M SwiGLU MLP blocks, GQA Attention blocks, and golden model differential tests.
"""

from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn.functional as F

from eswin_simulator.loader import LoadableModel
from eswin_simulator.memory.address_space import VirtualAddressSpace
from eswin_simulator.engines import (
    CMACEngine,
    CACCEngine,
    SDPEngine,
    PDPEngine,
    RubikEngine,
    SDPLookupTable,
    unpack_nvdla_weights_conv2d,
    unpack_nvdla_weights_1x1,
)
from eswin_simulator.dsp import DSPExecutor
from eswin_simulator.evaluation import TensorComparator


class TestEndToEndPipeline:
    @pytest.fixture
    def setup_engines(self):
        vas = VirtualAddressSpace()
        return {
            "vas": vas,
            "cmac": CMACEngine(vas),
            "cacc": CACCEngine(vas),
            "sdp": SDPEngine(vas),
            "pdp": PDPEngine(vas),
            "rubik": RubikEngine(vas),
            "dsp": DSPExecutor(vas),
        }

    def test_smollm_swiglu_mlp_block_pipeline(self, setup_engines):
        """Simulates full SmolLM-135M Transformer MLP block."""
        eng = setup_engines
        cmac, cacc, sdp, dsp = eng["cmac"], eng["cacc"], eng["sdp"], eng["dsp"]

        batch, seq, dim, hidden = 1, 4, 576, 1536
        np.random.seed(42)

        x = np.random.randn(batch, seq, dim).astype(np.float32) * 0.1
        w_norm = np.ones(dim, dtype=np.float32)
        w_gate = np.random.randn(hidden, dim).astype(np.float32) * 0.02
        w_up = np.random.randn(hidden, dim).astype(np.float32) * 0.02
        w_down = np.random.randn(dim, hidden).astype(np.float32) * 0.02

        # 1. DSP RMSNorm
        x_norm = dsp.execute_rms_norm(x, w_norm, eps=1e-5)

        # 2. CMAC Gate & Up Projections
        gate = cmac.gemm_1x1(x_norm, w_gate)
        up = cmac.gemm_1x1(x_norm, w_up)

        # 3. SDP SiLU Non-linear LUT activation
        silu_lut = SDPLookupTable.create_silu_lut()
        gate_act = sdp.interpolate_lut(gate, silu_lut)

        # 4. SDP Elementwise Gated Multiplication
        gated = sdp.eltwise_mul(gate_act, up)

        # 5. CMAC Down Projection
        down = cmac.gemm_1x1(gated, w_down)

        # 6. SDP Residual Add
        out_sim = sdp.eltwise_add(x, down)

        # PyTorch Reference Verification
        x_pt = torch.from_numpy(x)
        w_norm_pt = torch.from_numpy(w_norm)
        w_gate_pt = torch.from_numpy(w_gate)
        w_up_pt = torch.from_numpy(w_up)
        w_down_pt = torch.from_numpy(w_down)

        x_norm_pt = F.rms_norm(x_pt, (dim,), w_norm_pt, eps=1e-5)
        gate_pt = F.linear(x_norm_pt, w_gate_pt)
        up_pt = F.linear(x_norm_pt, w_up_pt)
        gated_pt = F.silu(gate_pt) * up_pt
        down_pt = F.linear(gated_pt, w_down_pt)
        out_pt = x_pt + down_pt

        rep = TensorComparator.compare(out_sim, out_pt.numpy(), tolerance=5e-4, cos_threshold=0.9999)
        assert rep.passed, f"MLP pipeline mismatch:\n{rep.summary()}"
        assert rep.cosine_similarity > 0.99999

    def test_smollm_attention_block_pipeline(self, setup_engines):
        """Simulates full SmolLM-135M Transformer Attention block with GQA & RoPE."""
        eng = setup_engines
        cmac, sdp, dsp, rubik = eng["cmac"], eng["sdp"], eng["dsp"], eng["rubik"]

        batch, seq, dim = 1, 8, 576
        num_q_heads, num_kv_heads, head_dim = 9, 3, 64
        np.random.seed(123)

        x = np.random.randn(batch, seq, dim).astype(np.float32) * 0.1
        w_norm = np.ones(dim, dtype=np.float32)
        wq = np.random.randn(dim, dim).astype(np.float32) * 0.02
        wk = np.random.randn(num_kv_heads * head_dim, dim).astype(np.float32) * 0.02
        wv = np.random.randn(num_kv_heads * head_dim, dim).astype(np.float32) * 0.02
        wo = np.random.randn(dim, dim).astype(np.float32) * 0.02

        # 1. RMSNorm
        x_norm = dsp.execute_rms_norm(x, w_norm, eps=1e-5)

        # 2. CMAC QKV Projections
        q_proj = cmac.gemm_1x1(x_norm, wq)
        k_proj = cmac.gemm_1x1(x_norm, wk)
        v_proj = cmac.gemm_1x1(x_norm, wv)

        # 3. Rubik Transpose to (B, H, S, D)
        q = rubik.transpose(rubik.reshape(q_proj, (batch, seq, num_q_heads, head_dim)), (0, 2, 1, 3))
        k = rubik.transpose(rubik.reshape(k_proj, (batch, seq, num_kv_heads, head_dim)), (0, 2, 1, 3))
        v = rubik.transpose(rubik.reshape(v_proj, (batch, seq, num_kv_heads, head_dim)), (0, 2, 1, 3))

        # 4. DSP RoPE
        q_rot = dsp.execute_rope(q, mode="half")
        k_rot = dsp.execute_rope(k, mode="half")

        # 5. GQA KV head repeat (3 -> 9)
        k_rep = np.repeat(k_rot, num_q_heads // num_kv_heads, axis=1)
        v_rep = np.repeat(v, num_q_heads // num_kv_heads, axis=1)

        # 6. Attention Scaled Dot-Product & Softmax
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.matmul(q_rot, np.transpose(k_rep, (0, 1, 3, 2))) * scale

        causal_mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
        scores_masked = scores + causal_mask

        probs = dsp.execute_softmax(scores_masked, axis=-1)
        context = np.matmul(probs, v_rep)

        # 7. Rubik Merge Heads
        context_merged = rubik.flatten(rubik.transpose(context, (0, 2, 1, 3)), start_dim=2, end_dim=3)

        # 8. CMAC Output Projection
        attn_out = cmac.gemm_1x1(context_merged, wo)

        # 9. SDP Residual Add
        out_sim = sdp.eltwise_add(x, attn_out)

        assert out_sim.shape == (batch, seq, dim)
        assert not np.any(np.isnan(out_sim))

        # Verify operation counts
        assert dsp.total_dsp_ops >= 3
        assert cmac.total_gemms >= 4
        assert rubik.total_reorders >= 5
        assert sdp.total_eltwise >= 1

    def test_golden_model_differential_with_comparator(self):
        """Verifies test_swiglu_mlp_npu_b1.model using TensorComparator against vendor output."""
        model_path = Path("/home/chri/aixigua/deep/artifacts/test_swiglu_mlp_npu_b1.model")
        golden_path = Path("/tmp/es_swiglu_golden.bin")
        if not golden_path.exists():
            golden_path = Path("/home/chri/aixigua/deep/artifacts/scratch/golden_swiglu/es_swiglu_golden.bin")

        if not (model_path.exists() and golden_path.exists()):
            pytest.skip("Golden model artifacts not present")

        model = LoadableModel.from_file(str(model_path))
        vas = VirtualAddressSpace(model)
        cmac = CMACEngine(vas)
        cacc = CACCEngine(vas)
        sdp = SDPEngine(vas)

        c_in = 576
        hidden_dim = 1536

        w_gate = np.vstack([
            unpack_nvdla_weights_1x1(model.blobs[f"w_gatesplitM_{i}_{i}"].data, 512, c_in, dtype=np.float16)
            for i in range(3)
        ])
        w_up = np.vstack([
            unpack_nvdla_weights_1x1(model.blobs[f"w_upsplitM_{i}_{i+3}"].data, 512, c_in, dtype=np.float16)
            for i in range(3)
        ])
        w_down = np.vstack([
            unpack_nvdla_weights_1x1(model.blobs["w_downsplitM_0_6"].data, 256, hidden_dim, dtype=np.float16),
            unpack_nvdla_weights_1x1(model.blobs["w_downsplitM_1_7"].data, 256, hidden_dim, dtype=np.float16),
            unpack_nvdla_weights_1x1(model.blobs["w_downsplitM_2_8"].data, 64, hidden_dim, dtype=np.float16),
        ])

        x = np.ones((1, 576, 1, 1), dtype=np.float16)

        gate_act = cmac.gemm_1x1(x, w_gate, precision="FP16")
        up_act = cmac.gemm_1x1(x, w_up, precision="FP16")

        lut_blob = model.blobs.get("kmd-task-lut_list")
        assert lut_blob is not None
        lut_desc = SDPLookupTable.from_bytes(lut_blob.data)
        silu_act = sdp.silu(gate_act, lut=lut_desc, use_lut=True)

        inter_act = sdp.eltwise_mul(silu_act, up_act)
        down_act = cmac.gemm_1x1(inter_act, w_down, precision="FP16")
        out_sim = cacc.finalize_fp16(down_act, to_fp16=True)

        golden = np.fromfile(str(golden_path), dtype=np.float16).reshape(1, 576, 1, 1)

        rep = TensorComparator.compare(out_sim, golden, tolerance=1e-3, cos_threshold=0.9999)
        assert rep.passed
        assert rep.cosine_similarity > 0.99999
        assert rep.max_abs_diff < 0.001

    def test_golden_qkv_proj_differential_with_comparator(self):
        """Verifies test_qkv_proj_npu_b1.model against vendor EsSimulator golden output."""
        model_path = Path("/home/chri/aixigua/deep/artifacts/test_qkv_proj_npu_b1.model")
        golden_dir = Path("/home/chri/aixigua/deep/artifacts/scratch/golden_qkv")

        if not (model_path.exists() and golden_dir.exists() and (golden_dir / "es_q_golden.bin").exists()):
            pytest.skip("Golden QKV artifacts not present")

        model = LoadableModel.from_file(str(model_path))
        vas = VirtualAddressSpace(model)
        cmac = CMACEngine(vas)
        cacc = CACCEngine(vas)

        x = np.fromfile(str(golden_dir / "in_qkv.bin"), dtype=np.float16).reshape(1, 576, 1, 1)

        wq = unpack_nvdla_weights_1x1(model.blobs["wq_0"].data, c_out=576, c_in=576, dtype=np.float16)
        wk = unpack_nvdla_weights_1x1(model.blobs["wk_1"].data, c_out=192, c_in=576, dtype=np.float16)
        wv = unpack_nvdla_weights_1x1(model.blobs["wv_2"].data, c_out=192, c_in=576, dtype=np.float16)

        q_sim = cacc.finalize_fp16(cmac.gemm_1x1(x, wq, precision="FP16"), to_fp16=True)
        k_sim = cacc.finalize_fp16(cmac.gemm_1x1(x, wk, precision="FP16"), to_fp16=True)
        v_sim = cacc.finalize_fp16(cmac.gemm_1x1(x, wv, precision="FP16"), to_fp16=True)

        q_gold = np.fromfile(str(golden_dir / "es_q_golden.bin"), dtype=np.float16).reshape(1, 576, 1, 1)
        k_gold = np.fromfile(str(golden_dir / "es_k_golden.bin"), dtype=np.float16).reshape(1, 192, 1, 1)
        v_gold = np.fromfile(str(golden_dir / "es_v_golden.bin"), dtype=np.float16).reshape(1, 192, 1, 1)

        rep_q = TensorComparator.compare(q_sim, q_gold, tolerance=1e-3, cos_threshold=0.9999)
        rep_k = TensorComparator.compare(k_sim, k_gold, tolerance=1e-3, cos_threshold=0.9999)
        rep_v = TensorComparator.compare(v_sim, v_gold, tolerance=1e-3, cos_threshold=0.9999)

        assert rep_q.passed
        assert rep_k.passed
        assert rep_v.passed
        assert rep_q.cosine_similarity > 0.99999
        assert rep_k.cosine_similarity > 0.99999
        assert rep_v.cosine_similarity > 0.99999

    def test_qwen2_5_coder_transformer_block_pipeline(self, setup_engines):
        """Simulates full Qwen2.5-Coder-1.5B Transformer block with GQA, RoPE (theta=1M), and SwiGLU MLP."""
        eng = setup_engines
        cmac, sdp, dsp, rubik = eng["cmac"], eng["sdp"], eng["dsp"], eng["rubik"]

        batch, seq = 1, 4
        dim = 1536              # Qwen2.5-Coder-1.5B hidden_size
        num_q_heads = 12        # 12 query heads
        num_kv_heads = 2        # 2 KV heads (GQA with ratio 6)
        head_dim = 128          # head dimension (12 * 128 = 1536)
        hidden_dim = 8960       # intermediate_size (SwiGLU)
        rope_theta = 1000000.0  # Qwen2.5 1M context theta
        rms_norm_eps = 1e-6

        np.random.seed(42)
        x = np.random.randn(batch, seq, dim).astype(np.float32) * 0.05

        # Attention weights
        w_norm1 = np.ones(dim, dtype=np.float32)
        wq = np.random.randn(num_q_heads * head_dim, dim).astype(np.float32) * 0.02
        wk = np.random.randn(num_kv_heads * head_dim, dim).astype(np.float32) * 0.02
        wv = np.random.randn(num_kv_heads * head_dim, dim).astype(np.float32) * 0.02
        wo = np.random.randn(dim, num_q_heads * head_dim).astype(np.float32) * 0.02

        # MLP weights
        w_norm2 = np.ones(dim, dtype=np.float32)
        w_gate = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        w_up = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.02
        w_down = np.random.randn(dim, hidden_dim).astype(np.float32) * 0.02

        # ---------------------------------------------------------------------
        # 1. Attention Block Execution via Simulator Engines
        # ---------------------------------------------------------------------
        # Pre-attention RMSNorm via DSP
        x_norm1 = dsp.execute_rms_norm(x, w_norm1, eps=rms_norm_eps)

        # Q, K, V Projections via CMAC
        q_proj = cmac.gemm_1x1(x_norm1, wq)
        k_proj = cmac.gemm_1x1(x_norm1, wk)
        v_proj = cmac.gemm_1x1(x_norm1, wv)

        # Rubik Reshape & Transpose to (B, H, S, D)
        q = rubik.transpose(rubik.reshape(q_proj, (batch, seq, num_q_heads, head_dim)), (0, 2, 1, 3))
        k = rubik.transpose(rubik.reshape(k_proj, (batch, seq, num_kv_heads, head_dim)), (0, 2, 1, 3))
        v = rubik.transpose(rubik.reshape(v_proj, (batch, seq, num_kv_heads, head_dim)), (0, 2, 1, 3))

        # RoPE via DSP with theta=1,000,000
        q_rot = dsp.execute_rope(q, theta_base=rope_theta, mode="half")
        k_rot = dsp.execute_rope(k, theta_base=rope_theta, mode="half")

        # GQA KV Repeat: 2 heads -> 12 heads (repeat 6x)
        k_rep = np.repeat(k_rot, num_q_heads // num_kv_heads, axis=1)
        v_rep = np.repeat(v, num_q_heads // num_kv_heads, axis=1)

        # Attention Scores & Softmax
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.matmul(q_rot, np.transpose(k_rep, (0, 1, 3, 2))) * scale
        causal_mask = np.triu(np.full((seq, seq), -1e9, dtype=np.float32), k=1)
        scores_masked = scores + causal_mask
        probs = dsp.execute_softmax(scores_masked, axis=-1)

        # Context multiply
        context = np.matmul(probs, v_rep)

        # Rubik Merge Heads
        context_merged = rubik.flatten(rubik.transpose(context, (0, 2, 1, 3)), start_dim=2, end_dim=3)

        # Output projection via CMAC
        attn_out = cmac.gemm_1x1(context_merged, wo)

        # Residual connection 1 via SDP
        h1 = sdp.eltwise_add(x, attn_out)

        # ---------------------------------------------------------------------
        # 2. SwiGLU MLP Block Execution via Simulator Engines
        # ---------------------------------------------------------------------
        # Post-attention RMSNorm via DSP
        h1_norm = dsp.execute_rms_norm(h1, w_norm2, eps=rms_norm_eps)

        # Gate and Up projections via CMAC
        gate_proj = cmac.gemm_1x1(h1_norm, w_gate)
        up_proj = cmac.gemm_1x1(h1_norm, w_up)

        # SiLU activation on Gate via SDP LUT
        silu_lut = SDPLookupTable.create_silu_lut()
        gate_act = sdp.interpolate_lut(gate_proj, silu_lut)

        # Elementwise Gate * Up via SDP
        gated = sdp.eltwise_mul(gate_act, up_proj)

        # Down projection via CMAC
        down_proj = cmac.gemm_1x1(gated, w_down)

        # Residual connection 2 via SDP
        out_sim = sdp.eltwise_add(h1, down_proj)

        # ---------------------------------------------------------------------
        # 3. Golden Reference via PyTorch
        # ---------------------------------------------------------------------
        x_pt = torch.from_numpy(x)
        w_norm1_pt = torch.from_numpy(w_norm1)
        wq_pt, wk_pt, wv_pt, wo_pt = [torch.from_numpy(w) for w in (wq, wk, wv, wo)]
        w_norm2_pt = torch.from_numpy(w_norm2)
        w_gate_pt, w_up_pt, w_down_pt = [torch.from_numpy(w) for w in (w_gate, w_up, w_down)]

        # Attn
        x_n1_pt = F.rms_norm(x_pt, (dim,), w_norm1_pt, eps=rms_norm_eps)
        q_pt = F.linear(x_n1_pt, wq_pt).view(batch, seq, num_q_heads, head_dim).transpose(1, 2)
        k_pt = F.linear(x_n1_pt, wk_pt).view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)
        v_pt = F.linear(x_n1_pt, wv_pt).view(batch, seq, num_kv_heads, head_dim).transpose(1, 2)

        q_rot_pt = torch.from_numpy(q_rot)
        k_rot_pt = torch.from_numpy(k_rot)
        k_rep_pt = k_rot_pt.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
        v_rep_pt = v_pt.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

        scores_pt = torch.matmul(q_rot_pt, k_rep_pt.transpose(-1, -2)) * scale
        scores_pt = scores_pt + torch.from_numpy(causal_mask)
        probs_pt = F.softmax(scores_pt, dim=-1)
        ctx_pt = torch.matmul(probs_pt, v_rep_pt).transpose(1, 2).contiguous().view(batch, seq, dim)
        attn_out_pt = F.linear(ctx_pt, wo_pt)
        h1_pt = x_pt + attn_out_pt

        # MLP
        h1_n_pt = F.rms_norm(h1_pt, (dim,), w_norm2_pt, eps=rms_norm_eps)
        gate_pt = F.linear(h1_n_pt, w_gate_pt)
        up_pt = F.linear(h1_n_pt, w_up_pt)
        gated_pt = F.silu(gate_pt) * up_pt
        down_pt = F.linear(gated_pt, w_down_pt)
        out_pt = h1_pt + down_pt

        rep = TensorComparator.compare(out_sim, out_pt.numpy(), tolerance=2e-3, cos_threshold=0.9999)
        assert rep.passed, f"Qwen2.5-Coder block verification failed:\n{rep.summary()}"
        assert rep.cosine_similarity > 0.99999

#!/usr/bin/env python3
"""
Lightweight Vision Encoder & Preprocessor for Liquid LFM3-VL-2B and d1-2B.

Preprocesses arbitrary resolution images into [gh_blocks, gw_blocks, 2, 2, 1536] patches,
runs the 164M hybrid vision encoder ONNX model, and produces [num_image_tokens, 2048]
visual embeddings ready for injection into the LFM3 tri-hybrid backbone.
"""

import os
import sys
import base64
import io
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np
from PIL import Image
import onnxruntime as ort

class Lfm3VisionEncoder:
    """Hybrid 164M Vision Encoder for LFM3-VL-2B and d1-2B."""

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            # Check default paths
            candidates = [
                "/home/chri/aixigua/deep/models/LFM3-VL-2B-ONNX",
                "/home/chri/aixigua/deep/models/d1-2B-ONNX",
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_dir = c
                    break
        if model_dir is None or not os.path.exists(model_dir):
            raise FileNotFoundError(f"Vision model directory not found: {model_dir}")

        path_obj = Path(model_dir)
        if path_obj.is_file():
            self.onnx_path = path_obj
            self.model_dir = path_obj.parent.parent
        else:
            self.model_dir = path_obj
            onnx_candidates = [
                self.model_dir / "onnx" / "vision_encoder.onnx",
                self.model_dir / "onnx" / "vision_encoder_fp16.onnx",
                self.model_dir / "vision_encoder.onnx",
                self.model_dir / "vision_encoder_fp16.onnx",
            ]
            self.onnx_path = None
            for cand in onnx_candidates:
                if cand.exists():
                    self.onnx_path = cand
                    break
            if self.onnx_path is None:
                raise FileNotFoundError(f"No vision_encoder.onnx found in {self.model_dir}")

        print(f"[+] Loading Vision Encoder from {self.onnx_path}...")
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )

        # ImageNet normalization parameters
        self.mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def preprocess_image(
        self,
        image_input: Union[str, Image.Image, bytes],
        max_dim: int = 512,
        min_dim: int = 256
    ) -> Tuple[np.ndarray, int, int]:
        """
        Loads and patches an image into [gh_blocks, gw_blocks, 2, 2, 1536].
        Returns (pixel_values, grid_height_blocks, grid_width_blocks).
        """
        if isinstance(image_input, str):
            if image_input.startswith("data:image") and "base64," in image_input:
                b64_data = image_input.split("base64,")[1]
                img = Image.open(io.BytesIO(base64.b64decode(b64_data))).convert("RGB")
            elif os.path.exists(image_input):
                img = Image.open(image_input).convert("RGB")
            else:
                raise ValueError(f"Invalid image path or format: {image_input[:64]}")
        elif isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        w, h = img.size
        # Scale image while preserving aspect ratio such that dims are multiples of 32
        scale = min(max_dim / max(w, h), 1.0)
        target_w = max(min_dim, int(round((w * scale) / 32.0)) * 32)
        target_h = max(min_dim, int(round((h * scale) / 32.0)) * 32)
        target_w = min(target_w, max_dim)
        target_h = min(target_h, max_dim)

        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std  # [H, W, 3]
        arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]

        C, H, W = arr.shape
        patch_size = 16
        merge_size = 2
        temp_size = 2

        grid_h = H // patch_size
        grid_w = W // patch_size
        gh_blocks = grid_h // merge_size
        gw_blocks = grid_w // merge_size

        # Reshape: [3, gh_blocks, 2, 16, gw_blocks, 2, 16]
        patches = arr.reshape(C, gh_blocks, merge_size, patch_size, gw_blocks, merge_size, patch_size)
        # Permute to: [gh_blocks, gw_blocks, merge_h, merge_w, C, patch_h, patch_w]
        patches = np.transpose(patches, (1, 4, 2, 5, 0, 3, 6))

        # Replicate along temporal dimension (temp_size=2)
        patches = np.repeat(np.expand_dims(patches, axis=4), temp_size, axis=4)
        patches = patches.reshape(gh_blocks, gw_blocks, merge_size, merge_size, C * temp_size * patch_size * patch_size)
        return patches.astype(np.float32), gh_blocks, gw_blocks

    def encode_image(self, image_input: Union[str, Image.Image, bytes], max_dim: int = 512) -> np.ndarray:
        """
        Runs the full vision pipeline on an image.
        Returns:
            image_features: np.ndarray of shape [num_image_tokens, 2048] (float32)
        """
        pixel_values, gh_blocks, gw_blocks = self.preprocess_image(image_input, max_dim=max_dim)
        num_expected_tokens = gh_blocks * gw_blocks

        inputs = {"pixel_values": pixel_values}
        outputs = self.session.run(None, inputs)
        features = outputs[0]  # [num_image_tokens, 2048]
        if features.shape[0] != num_expected_tokens:
            print(f"[!] Warning: features shape {features.shape[0]} != expected {num_expected_tokens}")
        return features

def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
        out_bin = sys.argv[2] if len(sys.argv) >= 3 else "visual_embeddings.bin"
        encoder = Lfm3VisionEncoder()
        features = encoder.encode_image(image_path)
        features.tofile(out_bin)
        print(f"[+] Encoded {image_path} -> {out_bin} ({features.shape[0]} visual tokens, {features.nbytes} bytes)")
        return

    encoder = Lfm3VisionEncoder()
    # Test on a dummy image
    img = Image.new("RGB", (320, 240), color="green")
    features = encoder.encode_image(img)
    print(f"[+] Output features shape: {features.shape} (dtype={features.dtype})")
    print(f"[+] Features stats: mean={features.mean():.4f}, std={features.std():.4f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
smollm_v2.py - SmolLM-135M Interactive Chat Engine (Python version)
Powered by Open ESWIN NPU Runtime.
"""

import sys
import os
import argparse

# Allow importing local eswin_runtime package
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_PY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "eswin_runtime", "python"))
if os.path.exists(RUNTIME_PY_DIR):
    sys.path.insert(0, RUNTIME_PY_DIR)

from eswin_runtime import RuntimeContext, SamplerParams, DeviceMode, version

def format_chatml(user_text: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def main():
    parser = argparse.ArgumentParser(description=f"SmolLM-v2 Python Chat Engine (Runtime v{version()})")
    parser.add_argument("-m", "--model", default=SCRIPT_DIR, help="Path to model directory with manifest.json")
    parser.add_argument("-t", "--tokenizer", help="Explicit tokenizer.json path")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P threshold")
    parser.add_argument("--repeat-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per turn")
    parser.add_argument("--single-die", action="store_true", help="Force single-die execution")
    args = parser.parse_args()

    dev_mode = DeviceMode.DIE0 if args.single_die else DeviceMode.DUAL_DIE_AUTO

    print(f"Initializing Open ESWIN Runtime v{version()}...")
    with RuntimeContext() as ctx:
        print(f"Hardware dies available: {ctx.device_count}")
        with ctx.load_model(args.model, args.tokenizer, device_mode=dev_mode) as model:
            if model.info:
                print(f"Loaded: {model.info.name} ({model.info.architecture})")
                print(f"Config: {model.info.num_layers} layers, {model.info.num_heads} heads, vocab={model.info.vocab_size}")

            params = SamplerParams(
                temperature=args.temp,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
            )

            with model.create_session(max_new_tokens=args.max_tokens, sampler_params=params) as session:
                print("\nSmolLM-v2 Ready! Type your prompt or /reset, /exit.\n")
                while True:
                    try:
                        user_input = input("User > ").strip()
                    except (KeyboardInterrupt, EOFError):
                        print("\nExiting...")
                        break

                    if not user_input:
                        continue
                    if user_input in ("/exit", "/quit"):
                        break
                    if user_input in ("/reset", "/clear"):
                        session.reset()
                        print("[Context cleared]")
                        continue

                    prompt = format_chatml(user_input)
                    print("SmolLM > ", end="", flush=True)

                    def stream_cb(token_id, piece):
                        print(piece, end="", flush=True)
                        return True

                    session.generate(prompt, callback=stream_cb)
                    print()

                    if session.last_stats:
                        s = session.last_stats
                        print(f"[{s.prompt_tokens} prompt in {s.prompt_eval_ms:.1f}ms | {s.eval_tokens} tokens in {s.eval_ms:.1f}ms | {s.tokens_per_sec:.2f} t/s]")

if __name__ == "__main__":
    main()

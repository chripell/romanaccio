#!/usr/bin/env python3
"""
chat_repl.py - Interactive chat REPL using eswin_runtime Python bindings.
"""

import sys
import argparse
from eswin_runtime import RuntimeContext, SamplerParams, DeviceMode, version

def main():
    parser = argparse.ArgumentParser(description=f"ESWIN NPU Python Chat REPL (v{version()})")
    parser.add_argument("-m", "--model", required=True, help="Path to model directory or manifest.json")
    parser.add_argument("-t", "--tokenizer", help="Path to tokenizer.json or vocab file")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K threshold")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-P nucleus threshold")
    parser.add_argument("--repeat-penalty", type=float, default=1.15, help="Repetition penalty")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per generation turn")
    parser.add_argument("--single-die", action="store_true", help="Force single-die execution")
    args = parser.parse_args()

    dev_mode = DeviceMode.DIE0 if args.single_die else DeviceMode.DUAL_DIE_AUTO

    print(f"Initializing ESWIN NPU Runtime v{version()}...")
    with RuntimeContext() as ctx:
        print(f"Operational dies detected: {ctx.device_count}")

        with ctx.load_model(args.model, args.tokenizer, device_mode=dev_mode) as model:
            if model.info:
                print(f"Loaded: {model.info.name} ({model.info.architecture})")
                print(f"Dimensions: hidden={model.info.hidden_size}, layers={model.info.num_layers}, heads={model.info.num_heads}")

            params = SamplerParams(
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
            )

            with model.create_session(max_new_tokens=args.max_tokens, sampler_params=params) as session:
                print("\nReady! Type your message, or /reset, /exit.\n")
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
                        print("[Context reset]")
                        continue

                    print("Assistant > ", end="", flush=True)

                    def stream_cb(token_id, piece):
                        print(piece, end="", flush=True)
                        return True

                    session.generate(user_input, callback=stream_cb)
                    print()
                    if session.last_stats:
                        s = session.last_stats
                        print(f"[{s.prompt_tokens} prompt in {s.prompt_eval_ms:.1f}ms | {s.eval_tokens} tokens in {s.eval_ms:.1f}ms | {s.tokens_per_sec:.2f} t/s]")

if __name__ == "__main__":
    main()

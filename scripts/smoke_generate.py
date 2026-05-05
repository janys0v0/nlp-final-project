"""Smoke test: load a small Qwen model and generate a few completions.

Verifies the env (uv, torch, transformers, GPU/MPS) is working end-to-end.

Usage:
    uv run python scripts/smoke_generate.py
    uv run python scripts/smoke_generate.py --model Qwen/Qwen2.5-0.5B
    uv run python scripts/smoke_generate.py --max-new-tokens 128 --temperature 0.0
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import torch
import transformers


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def pick_device_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file. Default: results/smoke/<model-slug>-<timestamp>.log",
    )
    args = parser.parse_args()

    if args.log_file is None:
        slug = args.model.replace("/", "_")
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        args.log_file = f"results/smoke/{slug}-{ts}.log"
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)
    print(f"logging to {log_path}")

    torch.manual_seed(args.seed)

    device, dtype = pick_device_dtype()
    print(f"device={device}  dtype={dtype}  model={args.model}")

    t0 = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dtype, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"loaded in {time.time() - t0:.1f}s  params={n_params:.2f}B")

    prompts = [
        "The capital of France is",
        "Solve step by step: 17 * 23 = ",
        "def fibonacci(n):\n    ",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        n_new = out.shape[1] - inputs.input_ids.shape[1]
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n--- prompt: {prompt!r}")
        print(f"--- {n_new} new tokens in {elapsed:.2f}s ({n_new / max(elapsed, 1e-6):.1f} tok/s)")
        print(text)


if __name__ == "__main__":
    main()

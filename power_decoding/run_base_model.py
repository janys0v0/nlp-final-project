import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from power_sampling_common import (
    MODEL_REPOS,
    answers_match,
    encode_text_prompt,
    ensure_math500,
    format_prompt,
    infer_model_device,
    load_generation_model,
    load_math500,
    load_text_processor,
    normalize_math_answer,
    parse_answer,
    select_shard,
    set_seed,
    sync_cuda,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Base-model-only generation on MATH500.")
    parser.add_argument("--model-key", default="qwen3_8b")
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--data-path", default="MATH500.json")
    parser.add_argument("--no-cot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_path = ensure_math500(args.data_path)
    dataset = load_math500(data_path)
    start, end, shard = select_shard(dataset, args.batch_idx, args.max_problems)

    model_str = MODEL_REPOS[args.model_key]
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"shard [{start}, {end}) of MATH500, model={model_str} device={device_name}")
    print("Loading tokenizer and model...")
    processor, tokenizer = load_text_processor(model_str)
    hf_model = load_generation_model(model_str)
    device = infer_model_device(hf_model)
    print("Model loaded.")

    p = next(hf_model.parameters())
    print(f"[diag] dtype={p.dtype} device={p.device} attn_impl={getattr(hf_model.config, '_attn_implementation', '?')}")
    if torch.cuda.is_available():
        print(f"[diag] gpu={torch.cuda.get_device_name(0)} cuda={torch.version.cuda}")

    do_sample = args.temperature > 0
    results = []
    for data in tqdm(shard, desc="MATH base model"):
        question = data["prompt"]
        answer = data["answer"]
        input_text = format_prompt(question, args.model_key, tokenizer, not args.no_cot)
        model_inputs = encode_text_prompt(processor, tokenizer, input_text, device)
        prompt_len = model_inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "return_dict_in_generate": True,
            "output_scores": False,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature

        sync_cuda(device)
        start_time = time.perf_counter()
        output = hf_model.generate(**model_inputs, **generate_kwargs)
        sync_cuda(device)
        seconds = time.perf_counter() - start_time

        output_ids = output.sequences[0, prompt_len:].to("cpu")
        completion = tokenizer.decode(output_ids, skip_special_tokens=True)
        parsed_answer = parse_answer(completion)
        normalized_answer = normalize_math_answer(parsed_answer)
        correct = answers_match(parsed_answer, answer)
        hit_eos = tokenizer.eos_token_id in output_ids.tolist()

        row = {
            "question": question,
            "correct_answer": answer,
            "correct_answer_normalized": normalize_math_answer(answer),
            "base_completion": completion,
            "base_answer": parsed_answer,
            "base_answer_normalized": normalized_answer,
            "base_correct": correct,
            "base_tokens": len(output_ids),
            "base_seconds": seconds,
            "base_tokens_per_second": len(output_ids) / max(seconds, 1e-9),
            "base_temperature": args.temperature,
            "base_do_sample": do_sample,
            "base_hit_eos": hit_eos,
            "base_truncated": not hit_eos,
            # Compatibility with the notebook baseline block.
            "naive_completion": completion,
            "naive_answer": parsed_answer,
            "naive_answer_normalized": normalized_answer,
            "naive_correct": correct,
            "naive_tokens": len(output_ids),
            "naive_seconds": seconds,
            "naive_tokens_per_second": len(output_ids) / max(seconds, 1e-9),
        }
        results.append(row)

    out_dir = Path(args.save_dir) / args.model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"{args.model_key}_math_base_temp{args.temperature}_batch{args.batch_idx}_seed{args.seed}.csv"
    )
    pd.DataFrame(results).to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()

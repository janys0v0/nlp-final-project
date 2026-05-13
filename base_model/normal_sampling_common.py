from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm


PROMPT = "Can you solve the following math problem? "
BASE = " Put your final answer within \\boxed{{}}."
COT = " Please reason step by step, and put your final answer within \\boxed{{}}."
GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: '\\boxed{{$LETTER}}' (without quotes) "
    "where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by step before answering.\n\n"
    "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)

# Kept in sync with other sampling modules.
MODEL_REPOS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_small": "Qwen/Qwen2.5-0.5B",
    "qwen_instruct_small": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen_math_small": "Qwen/Qwen2.5-Math-1.5B",
    "qwen3_small": "Qwen/Qwen3-0.6B",
    "qwen3_8b": "Qwen/Qwen3-8B",
    "qwen_math_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
}


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left) : -1]
    except Exception:
        return None


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def parse_answer(input_str):
    boxed = last_boxed_only_string(input_str or "")
    return remove_boxed(boxed) if boxed is not None else None


def normalize_math_answer(answer):
    if answer is None:
        return None
    s = str(answer).strip()
    boxed = last_boxed_only_string(s)
    if boxed is not None:
        s = remove_boxed(boxed) or s
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\!": "",
        "\\,": "",
        "\\;": "",
        "\\ ": "",
        "\\text": "",
        "\\mathrm": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return "".join(s.split()).strip(".$")


def answers_match(prediction, target):
    pred_norm = normalize_math_answer(prediction)
    target_norm = normalize_math_answer(target)
    return pred_norm is not None and target_norm is not None and pred_norm == target_norm


def normalize_choice_answer(answer):
    if answer is None:
        return None
    s = str(answer).strip().upper()
    boxed = last_boxed_only_string(s)
    if boxed is not None:
        s = remove_boxed(boxed) or s
        s = s.strip().upper()
    stripped = s.strip()
    if stripped in {"A", "B", "C", "D"}:
        return stripped
    matches = re.findall(r"\b([ABCD])\b", s)
    if matches:
        return matches[-1]
    return None


def choice_answers_match(prediction, target):
    pred_norm = normalize_choice_answer(prediction)
    target_norm = normalize_choice_answer(target)
    return pred_norm is not None and target_norm is not None and pred_norm == target_norm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_math500(path, repo="aakaran/reasoning-with-sampling", ref="main"):
    path = Path(path)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/llm_experiments/data/MATH500.json"
    print("Downloading", url)
    urllib.request.urlretrieve(url, path)
    print("Saved:", path, "size:", path.stat().st_size, "bytes")
    return path


def load_math500(path):
    with open(path, "r") as f:
        return json.load(f)


def load_json_dataset(path):
    with open(path, "r") as f:
        return json.load(f)


def select_shard(dataset, batch_idx, max_problems, shard_size=100):
    start = shard_size * batch_idx
    end = shard_size * (batch_idx + 1)
    if max_problems is not None:
        end = min(start + int(max_problems), end)
    return start, end, dataset[start:end]


def load_text_processor(model_str):
    processor = None
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_str, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return processor, tokenizer


def encode_text_prompt(processor, tokenizer, text, device):
    del processor
    return tokenizer(text, return_tensors="pt").to(device)


def load_generation_model(model_str):
    load_kwargs = {"torch_dtype": "auto", "device_map": "auto", "trust_remote_code": True}
    return transformers.AutoModelForCausalLM.from_pretrained(model_str, **load_kwargs).eval()


def infer_model_device(model):
    return next(model.parameters()).device


def sync_cuda(device):
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def format_prompt(question, model_key, tokenizer, cot=True):
    # Matches the formatting used elsewhere in this repo.
    if model_key in ("qwen", "qwen_small", "qwen_math", "qwen_math_small"):
        format_str = PROMPT + question
        format_str += COT if cot else BASE
    else:
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return format_str


def format_gpqa_prompt(example, model_key, tokenizer):
    prompt = GPQA_QUERY_TEMPLATE.format(
        Question=example["Question"],
        A=example["A"],
        B=example["B"],
        C=example["C"],
        D=example["D"],
    )
    if model_key in ("qwen", "qwen_small", "qwen_math", "qwen_math_small"):
        return prompt
    answer_context = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        answer_context,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def benchmark_question(example, benchmark):
    if benchmark == "math500":
        return example["prompt"]
    if benchmark == "gpqa":
        return example["Question"]
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def benchmark_answer(example, benchmark):
    if benchmark == "math500":
        return example["answer"]
    if benchmark == "gpqa":
        for key in ("Answer", "answer", "correct_answer"):
            if key in example:
                return example[key]
        raise KeyError("GPQA example is missing one of: Answer, answer, correct_answer")
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def normalize_benchmark_answer(answer, benchmark):
    if benchmark == "math500":
        return normalize_math_answer(answer)
    if benchmark == "gpqa":
        return normalize_choice_answer(answer)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def benchmark_answers_match(prediction, target, benchmark):
    if benchmark == "math500":
        return answers_match(prediction, target)
    if benchmark == "gpqa":
        return choice_answers_match(prediction, target)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


@dataclass
class NormalSamplingConfig:
    model_key: str = "qwen3_8b"
    batch_idx: int = 0
    seed: int = 0
    temperature: float = 0.1
    max_new_tokens: int = 1024
    max_problems: int | None = 10
    save_dir: str = "results"
    data_path: str = "MATH500.json"
    use_cot: bool = True
    greedy: bool = False


def run_normal_sampling(config: NormalSamplingConfig):
    set_seed(config.seed)

    data_path = ensure_math500(config.data_path)
    dataset = load_math500(data_path)
    start, end, shard = select_shard(dataset, config.batch_idx, config.max_problems)

    model_str = MODEL_REPOS[config.model_key]
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"shard [{start}, {end}) of MATH500, model={model_str} device={device_name}")
    print("Loading tokenizer and model...")
    processor, tokenizer = load_text_processor(model_str)
    hf_model = load_generation_model(model_str)
    device = infer_model_device(hf_model)
    print("Model loaded.")

    do_sample = (not config.greedy) and (config.temperature is not None) and (config.temperature > 0)
    effective_temperature = 0.0 if config.greedy else float(config.temperature)

    out_dir = Path(config.save_dir) / "normal_sampling"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for data in tqdm(shard, desc="MATH normal sampling"):
        question = data["prompt"]
        answer = data["answer"]
        input_text = format_prompt(question, config.model_key, tokenizer, config.use_cot)
        model_inputs = encode_text_prompt(processor, tokenizer, input_text, device)
        prompt_len = model_inputs["input_ids"].shape[1]

        generate_kwargs = {
            "max_new_tokens": int(config.max_new_tokens),
            "return_dict_in_generate": True,
            "output_scores": False,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = effective_temperature

        sync_cuda(device)
        t0 = time.perf_counter()
        output = hf_model.generate(**model_inputs, **generate_kwargs)
        sync_cuda(device)
        seconds = time.perf_counter() - t0

        output_ids = output.sequences[0, prompt_len:].to("cpu")
        completion = tokenizer.decode(output_ids, skip_special_tokens=True)
        parsed_answer = parse_answer(completion)
        normalized_answer = normalize_math_answer(parsed_answer)
        correct = answers_match(parsed_answer, answer)
        hit_eos = tokenizer.eos_token_id in output_ids.tolist()

        results.append(
            {
                "question": question,
                "correct_answer": answer,
                "correct_answer_normalized": normalize_math_answer(answer),
                "normal_completion": completion,
                "normal_answer": parsed_answer,
                "normal_answer_normalized": normalized_answer,
                "normal_correct": correct,
                "normal_tokens": len(output_ids),
                "normal_seconds": seconds,
                "normal_tokens_per_second": len(output_ids) / max(seconds, 1e-9),
                "normal_temperature": effective_temperature,
                "normal_do_sample": do_sample,
                "normal_greedy": bool(config.greedy),
                "normal_hit_eos": hit_eos,
                "normal_truncated": not hit_eos,
            }
        )

    df = pd.DataFrame(results)
    csv_name = (
        f"normal_{config.model_key}_temp{effective_temperature}_"
        f"maxnew{config.max_new_tokens}_batch{config.batch_idx}_seed{config.seed}.csv"
    )
    out_path = out_dir / csv_name
    df.to_csv(out_path, index=False)
    print("Saved:", out_path, "rows:", len(df))
    return df

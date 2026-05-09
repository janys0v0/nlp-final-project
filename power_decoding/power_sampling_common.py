from __future__ import annotations

import json
import os
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
from torch.nn import functional as F
from tqdm.auto import tqdm


PROMPT = "Can you solve the following math problem? "
BASE = " Put your final answer within \\boxed{{}}."
COT = " Please reason step by step, and put your final answer within \\boxed{{}}."
GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: '\\boxed{{$LETTER}}' (without quotes) where LETTER is one of ABCD (ex. '\\boxed{{A}}'). Think step by "
    "step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)

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


def parse_gpqa_answer(input_str):
    boxed = parse_answer(input_str or "")
    text = boxed if boxed is not None else input_str or ""
    matches = re.findall(r"\b(A|B|C|D)\b", str(text).upper())
    return matches[-1] if matches else None


def normalize_gpqa_answer(answer):
    if answer is None:
        return None
    matches = re.findall(r"\b(A|B|C|D)\b", str(answer).upper())
    return matches[-1] if matches else None


def normalize_benchmark_answer(dataset_name, answer):
    if dataset_name == "math":
        return normalize_math_answer(answer)
    if dataset_name == "gpqa":
        return normalize_gpqa_answer(answer)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def parse_benchmark_answer(dataset_name, completion):
    if dataset_name == "math":
        return parse_answer(completion)
    if dataset_name == "gpqa":
        return parse_gpqa_answer(completion)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def benchmark_answers_match(dataset_name, prediction, target):
    pred_norm = normalize_benchmark_answer(dataset_name, prediction)
    target_norm = normalize_benchmark_answer(dataset_name, target)
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
    path = Path(path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse MATH500 JSON file: {path}. "
            "If this is a GPQA data file, pass --dataset gpqa."
        ) from exc


def load_gpqa(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"GPQA data file not found: {path}. Pass --data-path pointing to a GPQA .jsonl, .json, or .csv file."
        )
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    raise ValueError(f"Unsupported GPQA data file extension: {path.suffix}")


def select_examples(dataset, max_problems):
    end = len(dataset) if max_problems is None else min(int(max_problems), len(dataset))
    return 0, end, dataset[:end]


def default_data_path(dataset_name):
    if dataset_name == "math":
        return "MATH500.json"
    if dataset_name == "gpqa":
        return "GPQA.jsonl"
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_benchmark_dataset(dataset_name, data_path):
    data_path = data_path or default_data_path(dataset_name)
    if dataset_name == "math":
        data_path = ensure_math500(data_path)
        return data_path, load_math500(data_path)
    if dataset_name == "gpqa":
        return Path(data_path), load_gpqa(data_path)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


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


def model_vocab_size(model):
    return model.get_input_embeddings().num_embeddings


def print_cuda_memory(label):
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{label}: cuda allocated={allocated:.2f} GiB reserved={reserved:.2f} GiB")


def forward_text_only(model, input_ids, attention_mask):
    return model(input_ids=input_ids, attention_mask=attention_mask)


def generate_text_only(model, input_ids, attention_mask, **generate_kwargs):
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)


def format_prompt(question, model_key, tokenizer, cot=True):
    if model_key in ("qwen", "qwen_small", "qwen_math", "qwen_math_small"):
        format_str = PROMPT + question
        format_str += COT if cot else BASE
    elif model_key in (
        "qwen_instruct_small",
        "qwen3_small",
        "qwen3_8b",
        "qwen_math_grpo",
        "phi_grpo",
        "phi",
        "tulu",
    ):
        content_str = PROMPT + question
        content_str += COT if cot else BASE
        answer_context = [{"role": "user", "content": content_str}]
        format_str = tokenizer.apply_chat_template(
            answer_context,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        raise ValueError(f"Unsupported model key for format_prompt: {model_key}")
    return format_str


def format_gpqa_prompt(data, seed):
    rng = random.Random(seed)
    choices = [
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    rng.shuffle(choices)
    gold_index = rng.randint(0, 3)
    choices.insert(gold_index, data["Correct Answer"])
    prompt = GPQA_QUERY_TEMPLATE.format(
        A=choices[0],
        B=choices[1],
        C=choices[2],
        D=choices[3],
        Question=data["Question"],
    )
    return prompt, "ABCD"[gold_index]


def prepare_benchmark_example(dataset_name, data, model_key, tokenizer, use_cot, example_seed):
    if dataset_name == "math":
        question = data["prompt"]
        answer = data["answer"]
        input_text = format_prompt(question, model_key, tokenizer, use_cot)
        return question, answer, input_text
    if dataset_name == "gpqa":
        input_text, answer = format_gpqa_prompt(data, example_seed)
        return input_text, answer, input_text
    raise ValueError(f"Unsupported dataset: {dataset_name}")


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        text_config = getattr(self.model.config, "text_config", self.model.config)
        self.block_size = getattr(text_config, "max_position_embeddings", 32768)

    @torch.no_grad()
    def next_token(self, prefix):
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(torch_prefix)
        prefix_cond = (
            torch_prefix
            if torch_prefix.size(1) <= self.block_size
            else torch_prefix[:, -self.block_size :]
        )
        mask_cond = attention_mask[:, -prefix_cond.size(1) :]
        output = forward_text_only(self.model, prefix_cond, mask_cond)
        logits = output.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)


def naive_temp(p, context, temp, seq_len):
    c = len(context)
    device = p.device
    tokenizer = p.tokenizer
    input_ids = torch.tensor([context], dtype=torch.long, device=device)
    output = generate_text_only(
        p.model,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    prop = output.sequences[0].tolist()
    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]
    idx = tokens.view(unscaled_logits.shape[0], 1, 1)
    log_probs_unnorm = (
        1 / temp * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)
    ).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()
    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)
    return prop, log_probs_norm, log_probs_unnorm


def mcmc_power_samp(p, context, temp, mcmc_steps, max_new_tokens, block_num=16, local_moves=False, local_window_blocks=1):
    c = len(context)
    print("alpha:", 1 / temp)
    gen = context.copy() if context is not None else []
    log_probs_norm = []
    log_probs_unnorm = []
    assert max_new_tokens % block_num == 0
    assert local_window_blocks >= 1
    assert not local_moves or local_window_blocks <= block_num
    jump_size = int(max_new_tokens // block_num)
    effective_window_blocks = local_window_blocks if local_moves else block_num
    recent_window_tokens = effective_window_blocks * jump_size
    print("max_new_tokens:", max_new_tokens, "jump_size:", jump_size, "local_moves:", local_moves, "local_window_blocks:", local_window_blocks, "recent_window_tokens:", recent_window_tokens)
    attempts = 0
    acceptances = 0
    block_stats = []
    for block_idx in tqdm(range(block_num)):
        sync_cuda(p.device)
        block_start = time.perf_counter()
        block_attempts = 0
        block_acceptances = 0
        gen, lp_norm, lp_unnorm = naive_temp(p, gen, temp=temp, seq_len=jump_size + len(gen))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        for _ in tqdm(range(mcmc_steps)):
            attempts += 1
            block_attempts += 1
            t = len(gen)
            lo = max(c, t - recent_window_tokens) if local_moves else c
            idx = random.randint(lo, t - 1)
            prop, log_prob_prop, target_log_prob_prop = naive_temp(p, gen[:idx], temp=temp, seq_len=t)
            s = len(prop)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx
            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]
            log_r = (
                sum(target_log_prob_prop)
                + sum(log_prob_cur)
                - sum(target_log_prob_cur)
                - sum(log_prob_prop)
            )
            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                block_acceptances += 1
                gen = prop.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()
        sync_cuda(p.device)
        block_stats.append({
            "block_idx": block_idx,
            "block_attempts": block_attempts,
            "block_acceptances": block_acceptances,
            "block_acceptance_rate": block_acceptances / max(1, block_attempts),
            "block_seconds": time.perf_counter() - block_start,
            "recent_window_tokens": recent_window_tokens,
        })
        if p.tokenizer.eos_token_id in gen[c:]:
            eos_idx = c + gen[c:].index(p.tokenizer.eos_token_id)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx - c + 1]
            log_probs_unnorm = log_probs_unnorm[: eos_idx - c + 1]
            return gen, log_probs_norm, log_probs_unnorm, acceptances / max(1, attempts), block_stats
    return gen, log_probs_norm, log_probs_unnorm, acceptances / max(1, attempts), block_stats


def sync_cuda(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


@dataclass
class PowerExperimentConfig:
    model_key: str = "qwen3_8b"
    dataset: str = "math"
    seed: int = 0
    mcmc_steps: int = 4
    temperature: float = 0.25
    max_new_tokens: int = 1024
    block_num: int = 16
    max_problems: int | None = 10
    save_dir: str = "results"
    data_path: str | None = None
    use_cot: bool = True
    local_moves: bool = True
    local_window_blocks: int = 1
    include_baselines: bool = True


def run_power_experiment(config: PowerExperimentConfig, experiment_name: str):
    set_seed(config.seed)
    dataset_name = config.dataset.lower()
    if dataset_name not in ("math", "gpqa"):
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    if config.max_new_tokens % config.block_num != 0:
        raise ValueError(f"max_new_tokens must be divisible by block_num: {config.max_new_tokens} % {config.block_num}")
    if config.local_window_blocks < 1:
        raise ValueError("local_window_blocks must be >= 1")
    if config.local_moves and config.local_window_blocks > config.block_num:
        raise ValueError("local_window_blocks cannot exceed block_num for local moves")
    jump_size = config.max_new_tokens // config.block_num
    effective_window_blocks = config.local_window_blocks if config.local_moves else config.block_num
    recent_window_tokens = effective_window_blocks * jump_size
    data_path, dataset = load_benchmark_dataset(dataset_name, config.data_path)
    start, end, shard = select_examples(dataset, config.max_problems)

    model_str = MODEL_REPOS[config.model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"examples [{start}, {end}) of {dataset_name.upper()}, model={model_str} device={device} data={data_path}")

    out_dir = Path(config.save_dir) / config.model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading tokenizer and model...")
    processor, tokenizer = load_text_processor(model_str)
    hf_model = load_generation_model(model_str)
    device = infer_model_device(hf_model)
    sampler = AutoregressiveSampler(hf_model, tokenizer, device)
    print("Model loaded.")

    p = next(hf_model.parameters())
    print(f"[diag] dtype={p.dtype} device={p.device} attn_impl={getattr(hf_model.config, '_attn_implementation', '?')}")
    if torch.cuda.is_available():
        print(f"[diag] gpu={torch.cuda.get_device_name(0)} cuda={torch.version.cuda}")

    results = []
    for problem_idx, data in tqdm(
        enumerate(shard, start=start),
        total=len(shard),
        desc=f"{dataset_name.upper()} {experiment_name} power sampling",
    ):
        question, answer, input_text = prepare_benchmark_example(
            dataset_name,
            data,
            config.model_key,
            tokenizer,
            config.use_cot,
            example_seed=config.seed + problem_idx,
        )
        model_inputs = encode_text_prompt(processor, tokenizer, input_text, device)
        input_ids = model_inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        prefix = [idx.item() for idx in input_ids[0]]

        row = {
            "dataset": dataset_name,
            "question": question,
            "correct_answer": answer,
            "correct_answer_normalized": normalize_benchmark_answer(dataset_name, answer),
        }

        if config.include_baselines:
            sync_cuda(device)
            naive_start = time.perf_counter()
            naive_output = hf_model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                temperature=config.temperature,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            sync_cuda(device)
            naive_seconds = time.perf_counter() - naive_start

            sync_cuda(device)
            std_start = time.perf_counter()
            std_output = hf_model.generate(
                **model_inputs,
                max_new_tokens=config.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            sync_cuda(device)
            std_seconds = time.perf_counter() - std_start

            naive_ids = naive_output.sequences[0, prompt_len:].to("cpu")
            std_ids = std_output.sequences[0, prompt_len:].to("cpu")
            naive_completion = tokenizer.decode(naive_ids, skip_special_tokens=True)
            std_completion = tokenizer.decode(std_ids, skip_special_tokens=True)
            naive_answer = parse_benchmark_answer(dataset_name, naive_completion)
            std_answer = parse_benchmark_answer(dataset_name, std_completion)

            row.update(
                {
                    "naive_completion": naive_completion,
                    "naive_answer": naive_answer,
                    "naive_answer_normalized": normalize_benchmark_answer(dataset_name, naive_answer),
                    "naive_correct": benchmark_answers_match(dataset_name, naive_answer, answer),
                    "naive_tokens": len(naive_ids),
                    "naive_seconds": naive_seconds,
                    "naive_tokens_per_second": len(naive_ids) / max(naive_seconds, 1e-9),
                    "std_completion": std_completion,
                    "std_answer": std_answer,
                    "std_answer_normalized": normalize_benchmark_answer(dataset_name, std_answer),
                    "std_correct": benchmark_answers_match(dataset_name, std_answer, answer),
                    "std_tokens": len(std_ids),
                    "std_seconds": std_seconds,
                    "std_tokens_per_second": len(std_ids) / max(std_seconds, 1e-9),
                }
            )

        sync_cuda(device)
        mcmc_start = time.perf_counter()
        mcmc_output, _, _, acceptance_ratio, block_stats = mcmc_power_samp(
            sampler,
            prefix,
            config.temperature,
            config.mcmc_steps,
            max_new_tokens=config.max_new_tokens,
            block_num=config.block_num,
            local_moves=config.local_moves,
            local_window_blocks=config.local_window_blocks,
        )
        sync_cuda(device)
        mcmc_seconds = time.perf_counter() - mcmc_start

        mcmc_ids = mcmc_output[prompt_len:]
        mcmc_completion = tokenizer.decode(mcmc_ids, skip_special_tokens=True)
        mcmc_answer = parse_benchmark_answer(dataset_name, mcmc_completion)
        mcmc_hit_eos = tokenizer.eos_token_id in mcmc_ids
        row.update(
            {
                "mcmc_completion": mcmc_completion,
                "mcmc_answer": mcmc_answer,
                "mcmc_answer_normalized": normalize_benchmark_answer(dataset_name, mcmc_answer),
                "mcmc_correct": benchmark_answers_match(dataset_name, mcmc_answer, answer),
                "mcmc_tokens": len(mcmc_ids),
                "mcmc_seconds": mcmc_seconds,
                "mcmc_tokens_per_second": len(mcmc_ids) / max(mcmc_seconds, 1e-9),
                "acceptance_ratio": acceptance_ratio,
                "mcmc_local_moves": config.local_moves,
                "mcmc_local_window_blocks": config.local_window_blocks,
                "mcmc_effective_window_blocks": effective_window_blocks,
                "mcmc_recent_window_tokens": recent_window_tokens,
                "mcmc_block_num": config.block_num,
                "mcmc_jump_size": jump_size,
                "mcmc_hit_eos": mcmc_hit_eos,
                "mcmc_truncated": not mcmc_hit_eos,
                "mcmc_blocks_completed": len(block_stats),
                "mcmc_block_acceptance_rates": json.dumps([b["block_acceptance_rate"] for b in block_stats]),
                "mcmc_block_seconds": json.dumps([b["block_seconds"] for b in block_stats]),
                "mcmc_block_recent_window_tokens": json.dumps([b["recent_window_tokens"] for b in block_stats]),
            }
        )
        results.append(row)
        print(
            "acceptance_ratio:",
            acceptance_ratio,
            "mcmc_correct:",
            benchmark_answers_match(dataset_name, mcmc_answer, answer),
        )

    df = pd.DataFrame(results)
    csv_name = (
        f"{config.model_key}_{dataset_name}_power_{experiment_name}_"
        f"steps{config.mcmc_steps}_blocks{config.block_num}_window{effective_window_blocks}_temp{config.temperature}_"
        f"n{len(shard)}_seed{config.seed}.csv"
    )
    out_path = out_dir / csv_name
    df.to_csv(out_path, index=False)
    metric_cols = [
        "mcmc_correct",
        "acceptance_ratio",
        "mcmc_tokens_per_second",
        "mcmc_seconds",
        "mcmc_tokens",
        "mcmc_recent_window_tokens",
    ]
    if config.include_baselines:
        metric_cols = [
            "naive_correct",
            "std_correct",
            *metric_cols,
            "naive_tokens_per_second",
            "std_tokens_per_second",
        ]
    print(df[metric_cols].mean(numeric_only=True).to_frame("mean").T)
    print("Saved:", out_path)
    return out_path


@torch.no_grad()
def score_continuation(model, context_ids, continuation_ids):
    if continuation_ids.numel() == 0:
        return torch.zeros(context_ids.shape[0], device=context_ids.device)
    full_ids = torch.cat([context_ids, continuation_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    outputs = forward_text_only(model, full_ids, attention_mask)
    start = context_ids.shape[1] - 1
    end = full_ids.shape[1] - 1
    pred_logits = outputs.logits[:, start:end, :]
    log_probs = F.log_softmax(pred_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(-1, continuation_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=1)


def _proposal_logprob_from_generate_scores(scores, continuation_ids):
    step_logps = []
    for step_scores, step_token in zip(scores, continuation_ids[0]):
        log_probs = F.log_softmax(step_scores.float(), dim=-1)
        step_logps.append(log_probs[0, step_token])
    if not step_logps:
        return torch.tensor(0.0, device=continuation_ids.device)
    return torch.stack(step_logps).sum()


@torch.no_grad()
def draft_propose(draft_model, tokenizer, context_ids, block_size, temperature=1.0):
    out = generate_text_only(
        draft_model,
        input_ids=context_ids,
        attention_mask=torch.ones_like(context_ids),
        max_new_tokens=block_size,
        min_new_tokens=1,
        do_sample=True,
        temperature=temperature,
        return_dict_in_generate=True,
        output_scores=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    continuation_ids = out.sequences[:, context_ids.shape[1] :]
    draft_logprob = _proposal_logprob_from_generate_scores(out.scores, continuation_ids)
    return continuation_ids, draft_logprob


def speculative_power_step(
    target_model,
    draft_model,
    tokenizer,
    context_ids,
    old_block_ids,
    old_target_logprob,
    old_draft_logprob,
    alpha,
    block_size,
    draft_temperature=1.0,
):
    new_block_ids, new_draft_logprob = draft_propose(
        draft_model, tokenizer, context_ids, block_size, temperature=draft_temperature
    )
    new_target_logprob = score_continuation(target_model, context_ids, new_block_ids).squeeze(0)
    log_accept_ratio = (
        alpha * new_target_logprob
        + old_draft_logprob
        - alpha * old_target_logprob
        - new_draft_logprob
    )
    log_accept_prob = torch.minimum(
        log_accept_ratio,
        torch.tensor(0.0, device=context_ids.device, dtype=log_accept_ratio.dtype),
    )
    accepted = torch.log(torch.rand((), device=context_ids.device)) < log_accept_prob
    if bool(accepted.item()):
        return new_block_ids, new_target_logprob, new_draft_logprob, True, float(log_accept_ratio.item())
    return old_block_ids, old_target_logprob, old_draft_logprob, False, float(log_accept_ratio.item())


def run_speculative_power_sampling(
    target_model,
    draft_model,
    tokenizer,
    context_ids,
    alpha,
    block_size,
    mcmc_steps_per_block,
    max_new_tokens,
    draft_temperature=1.0,
):
    generated = []
    context = context_ids
    block_stats = []
    total_attempts = 0
    total_acceptances = 0
    max_blocks = int(np.ceil(max_new_tokens / block_size))
    for block_idx in tqdm(range(max_blocks), desc="Speculative power blocks"):
        remaining = max_new_tokens - len(generated)
        current_block_size = min(block_size, remaining)
        if current_block_size <= 0:
            break
        old_block, old_draft_logprob = draft_propose(
            draft_model, tokenizer, context, current_block_size, temperature=draft_temperature
        )
        old_target_logprob = score_continuation(target_model, context, old_block).squeeze(0)
        block_acceptances = 0
        last_log_accept_ratio = None
        for _ in range(mcmc_steps_per_block):
            total_attempts += 1
            old_block, old_target_logprob, old_draft_logprob, accepted, last_log_accept_ratio = (
                speculative_power_step(
                    target_model=target_model,
                    draft_model=draft_model,
                    tokenizer=tokenizer,
                    context_ids=context,
                    old_block_ids=old_block,
                    old_target_logprob=old_target_logprob,
                    old_draft_logprob=old_draft_logprob,
                    alpha=alpha,
                    block_size=current_block_size,
                    draft_temperature=draft_temperature,
                )
            )
            if accepted:
                total_acceptances += 1
                block_acceptances += 1
        block_tokens = old_block[0].tolist()
        generated.extend(block_tokens)
        context = torch.cat([context, old_block], dim=1)
        block_stats.append(
            {
                "block_idx": block_idx,
                "block_tokens": len(block_tokens),
                "block_acceptance_rate": block_acceptances / max(1, mcmc_steps_per_block),
                "target_logprob": float(old_target_logprob.item()),
                "draft_logprob": float(old_draft_logprob.item()),
                "last_log_accept_ratio": last_log_accept_ratio,
            }
        )
        if tokenizer.eos_token_id in block_tokens:
            eos_pos = generated.index(tokenizer.eos_token_id)
            generated = generated[: eos_pos + 1]
            break
    return generated, {
        "acceptance_ratio": total_acceptances / max(1, total_attempts),
        "attempts": total_attempts,
        "acceptances": total_acceptances,
        "blocks": block_stats,
    }


@dataclass
class SpeculativeExperimentConfig:
    target_model_key: str = "qwen3_8b"
    draft_model_key: str = "qwen3_small"
    dataset: str = "math"
    seed: int = 0
    alpha: float = 4.0
    block_size: int = 64
    mcmc_steps_per_block: int = 4
    max_new_tokens: int = 1024
    draft_temperature: float = 1.0
    max_problems: int | None = 10
    save_dir: str = "results"
    data_path: str | None = None
    use_cot: bool = True


def run_speculative_experiment(config: SpeculativeExperimentConfig):
    set_seed(config.seed)
    dataset_name = config.dataset.lower()
    if dataset_name not in ("math", "gpqa"):
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    data_path, dataset = load_benchmark_dataset(dataset_name, config.data_path)
    start, end, shard = select_examples(dataset, config.max_problems)

    target_model_str = MODEL_REPOS[config.target_model_key]
    draft_model_str = MODEL_REPOS[config.draft_model_key]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"spec examples [{start}, {end}) of {dataset_name.upper()} "
        f"target={target_model_str} draft={draft_model_str} device={device} data={data_path}"
    )

    out_dir = Path(config.save_dir) / "speculative_power"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading shared tokenizer...")
    processor, tokenizer = load_text_processor(target_model_str)

    print("Loading target model...")
    target_model = load_generation_model(target_model_str)
    print_cuda_memory("after target load")

    print("Loading draft model...")
    draft_model = load_generation_model(draft_model_str)
    print_cuda_memory("after draft load")

    device = infer_model_device(target_model)
    target_vocab_size = model_vocab_size(target_model)
    draft_vocab_size = model_vocab_size(draft_model)
    tokenizer_size = len(tokenizer)

    if tokenizer_size > min(target_vocab_size, draft_vocab_size):
        raise ValueError(
            "Tokenizer can emit ids outside at least one model embedding matrix. "
            f"target={target_vocab_size}, draft={draft_vocab_size}, tokenizer={tokenizer_size}"
        )
    print("Models loaded.")

    results = []
    for problem_idx, data in tqdm(
        enumerate(shard, start=start),
        total=len(shard),
        desc=f"{dataset_name.upper()} speculative power",
    ):
        question, answer, input_text = prepare_benchmark_example(
            dataset_name,
            data,
            config.target_model_key,
            tokenizer,
            config.use_cot,
            example_seed=config.seed + problem_idx,
        )
        model_inputs = encode_text_prompt(processor, tokenizer, input_text, device)
        context_ids = model_inputs["input_ids"]

        sync_cuda(device)
        t0 = time.perf_counter()
        generated_ids, spec_stats = run_speculative_power_sampling(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            context_ids=context_ids,
            alpha=config.alpha,
            block_size=config.block_size,
            mcmc_steps_per_block=config.mcmc_steps_per_block,
            max_new_tokens=config.max_new_tokens,
            draft_temperature=config.draft_temperature,
        )
        sync_cuda(device)
        seconds = time.perf_counter() - t0

        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        predicted_answer = parse_benchmark_answer(dataset_name, completion)
        results.append(
            {
                "dataset": dataset_name,
                "question": question,
                "correct_answer": answer,
                "correct_answer_normalized": normalize_benchmark_answer(dataset_name, answer),
                "spec_completion": completion,
                "spec_answer": predicted_answer,
                "spec_answer_normalized": normalize_benchmark_answer(dataset_name, predicted_answer),
                "spec_correct": benchmark_answers_match(dataset_name, predicted_answer, answer),
                "spec_tokens": len(generated_ids),
                "spec_seconds": seconds,
                "spec_tokens_per_second": len(generated_ids) / max(seconds, 1e-9),
                "spec_acceptance_ratio": spec_stats["acceptance_ratio"],
                "spec_attempts": spec_stats["attempts"],
                "spec_acceptances": spec_stats["acceptances"],
            }
        )
        print(
            "spec_acceptance_ratio:",
            spec_stats["acceptance_ratio"],
            "correct:",
            benchmark_answers_match(dataset_name, predicted_answer, answer),
            "tokens/sec:",
            len(generated_ids) / max(seconds, 1e-9),
        )

    df = pd.DataFrame(results)
    csv_name = (
        f"spec_power_{dataset_name}_{config.draft_model_key}_to_{config.target_model_key}_"
        f"alpha{config.alpha}_block{config.block_size}_steps{config.mcmc_steps_per_block}_"
        f"n{len(shard)}_seed{config.seed}.csv"
    )
    out_path = out_dir / csv_name
    df.to_csv(out_path, index=False)
    metric_cols = [
        "spec_correct",
        "spec_acceptance_ratio",
        "spec_tokens_per_second",
        "spec_seconds",
        "spec_tokens",
    ]
    print(df[metric_cols].mean(numeric_only=True).to_frame("mean").T)
    print("Saved:", out_path)
    return out_path

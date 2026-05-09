import argparse

from power_sampling_common import SpeculativeExperimentConfig, run_speculative_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Speculative power decoding on MATH500 or GPQA.")
    parser.add_argument("--dataset", choices=("math", "gpqa"), default="math")
    parser.add_argument("--target-model-key", default="qwen3_8b")
    parser.add_argument("--draft-model-key", default="qwen3_small")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--block-size", type=int, default=192)
    parser.add_argument("--mcmc-steps-per-block", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--draft-temperature", type=float, default=1.0)
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--save-dir", default="results")
    parser.add_argument(
        "--data-path",
        default=None,
        help="Benchmark data path. Defaults to MATH500.json for math and GPQA.jsonl for gpqa.",
    )
    parser.add_argument("--no-cot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = SpeculativeExperimentConfig(
        target_model_key=args.target_model_key,
        draft_model_key=args.draft_model_key,
        dataset=args.dataset,
        seed=args.seed,
        alpha=args.alpha,
        block_size=args.block_size,
        mcmc_steps_per_block=args.mcmc_steps_per_block,
        max_new_tokens=args.max_new_tokens,
        draft_temperature=args.draft_temperature,
        max_problems=args.max_problems,
        save_dir=args.save_dir,
        data_path=args.data_path,
        use_cot=not args.no_cot,
    )
    run_speculative_experiment(config)


if __name__ == "__main__":
    main()

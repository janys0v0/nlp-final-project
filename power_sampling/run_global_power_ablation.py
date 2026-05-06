import argparse

from power_sampling_common import PowerExperimentConfig, run_power_experiment


MCMC_LOCAL_MOVES = False


def parse_args():
    parser = argparse.ArgumentParser(description="Global power-sampling ablation on MATH500.")
    parser.add_argument("--model-key", default="qwen3_8b")
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcmc-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--block-num", type=int, default=16)
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--save-dir", default="results")
    parser.add_argument("--data-path", default="MATH500.json")
    parser.add_argument("--no-cot", action="store_true")
    parser.add_argument("--skip-naive-std", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = PowerExperimentConfig(
        model_key=args.model_key,
        batch_idx=args.batch_idx,
        seed=args.seed,
        mcmc_steps=args.mcmc_steps,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        block_num=args.block_num,
        max_problems=args.max_problems,
        save_dir=args.save_dir,
        data_path=args.data_path,
        use_cot=not args.no_cot,
        local_moves=MCMC_LOCAL_MOVES,  # Global ablation: resample any suffix position.
        include_baselines=not args.skip_naive_std,
    )
    run_power_experiment(config, experiment_name="global")


if __name__ == "__main__":
    main()

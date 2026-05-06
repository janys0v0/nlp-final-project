import argparse

from power_sampling_common import PowerExperimentConfig, run_power_experiment


MCMC_LOCAL_MOVES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Local/windowed power-sampling ablation on MATH500.")
    parser.add_argument("--model-key", default="qwen3_8b")
    parser.add_argument("--batch-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcmc-steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--block-num", type=int, default=16)
    parser.add_argument(
        "--local-window-blocks",
        type=int,
        default=1,
        help="MCMC proposals resample within the most recent N blocks (N * jump_size tokens). Default 1.",
    )
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
        local_window_blocks=args.local_window_blocks,
        max_problems=args.max_problems,
        save_dir=args.save_dir,
        data_path=args.data_path,
        use_cot=not args.no_cot,
        local_moves=MCMC_LOCAL_MOVES,  # Local ablation: only resample within the latest jump-size window.
        include_baselines=not args.skip_naive_std,
    )
    run_power_experiment(config, experiment_name="local")


if __name__ == "__main__":
    main()

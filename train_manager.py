from argparse import ArgumentParser
from pathlib import Path

from core.train_manager import TrainManager


def parse_ids(raw):
    """Parse comma-separated run ids like '1,2,3'."""
    if raw is None or raw.strip() == "":
        return None
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))
    return out


def parse_range(raw):
    """Parse run range like '1-5'."""
    if raw is None or "-" not in raw:
        return None
    left, right = raw.split("-", 1)
    left, right = left.strip(), right.strip()
    if not left.isdigit() or not right.isdigit():
        return None
    start_id, end_id = int(left), int(right)
    lo, hi = min(start_id, end_id), max(start_id, end_id)
    return list(range(lo, hi + 1))


def build_parser():
    parser = ArgumentParser(description="Train run manager for baseline/vlm folders")
    parser.add_argument("--algo", required=True, choices=["baseline", "vlm"], help="Algorithm folder name")
    parser.add_argument("--base-dir", default="logs_and_results", help="Root folder for logs and results")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List existing runs")

    create_cmd = sub.add_parser("create", help="Create a new run folder")
    create_cmd.add_argument("--run-id", type=int, default=None, help="Optional fixed run id")

    summary_cmd = sub.add_parser("summary", help="Show mean/min/max of thresholds")
    summary_cmd.add_argument("--ids", default=None, help="Comma-separated run ids, example: 1,2,3")
    summary_cmd.add_argument("--range", dest="run_range", default=None, help="Range, example: 1-5")

    clean_cmd = sub.add_parser("clean", help="Delete runs")
    clean_cmd.add_argument("--ids", default=None, help="Comma-separated run ids, example: 1,2,3")
    clean_cmd.add_argument("--range", dest="run_range", default=None, help="Range, example: 1-5")

    return parser


def choose_ids(args):
    ids_from_range = parse_range(getattr(args, "run_range", None))
    ids_from_list = parse_ids(getattr(args, "ids", None))
    if ids_from_range is not None:
        return ids_from_range
    return ids_from_list


def main():
    parser = build_parser()
    args = parser.parse_args()

    manager = TrainManager(base_dir=Path(args.base_dir), algo_name=args.algo)

    if args.command == "list":
        run_ids = manager.list_run_ids()
        print(f"{args.algo} runs: {run_ids}")
        return

    if args.command == "create":
        run_dir, run_id = manager.create_run_dir(run_id=args.run_id)
        paths = manager.get_run_paths(run_id=run_id, create=False)
        print(f"Created run_{run_id}: {run_dir}")
        print(f"models: {paths['models_dir']}")
        print(f"videos: {paths['videos_dir']}")
        print(f"data: {paths['data_dir']}")
        print(f"tensorboard: {paths['tensorboard_dir']}")
        return

    run_ids = choose_ids(args)

    if args.command == "summary":
        summary = manager.summarize_results(run_ids=run_ids)
        if not summary:
            print("No threshold data found for selected runs.")
            return
        print("Threshold summary:")
        for threshold, stats in sorted(summary.items(), key=lambda kv: float(kv[0])):
            mean_v = stats["mean"]
            min_v = stats["min"]
            max_v = stats["max"]
            count_v = stats["count"]
            print(
                f"  threshold {threshold}: mean={mean_v:.2f}, "
                f"min={min_v}, max={max_v}, n={count_v}"
            )
        return

    if args.command == "clean":
        manager.clean_run_dirs(run_ids=run_ids)
        print("Clean completed.")
        return


if __name__ == "__main__":
    main()

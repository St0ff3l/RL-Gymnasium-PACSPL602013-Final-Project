import json
import shutil
from pathlib import Path

import pandas as pd


class TrainManager:
    """Manage run folders and training artifacts for one algorithm."""

    SUBDIRS = ("models", "videos", "data", "tensorboard")

    def __init__(self, base_dir="logs_and_results", algo_name="baseline"):
        self.algo_name = algo_name
        self.algo_dir = Path(base_dir) / algo_name
        self.algo_dir.mkdir(parents=True, exist_ok=True)

    def _parse_run_id(self, run_name):
        if not run_name.startswith("run_"):
            return None
        raw = run_name.split("_", 1)[1]
        return int(raw) if raw.isdigit() else None

    def list_run_ids(self):
        """Return sorted run ids like [1, 2, 3]."""
        run_ids = []
        for path in self.algo_dir.glob("run_*"):
            if not path.is_dir():
                continue
            rid = self._parse_run_id(path.name)
            if rid is not None:
                run_ids.append(rid)
        return sorted(run_ids)

    def get_next_run_id(self):
        run_ids = self.list_run_ids()
        return (max(run_ids) + 1) if run_ids else 1

    def get_run_dir(self, run_id):
        return self.algo_dir / f"run_{int(run_id)}"

    def create_run_dir(self, run_id=None):
        """Create run folder and standard subfolders."""
        if run_id is None:
            run_id = self.get_next_run_id()

        run_dir = self.get_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        for sub in self.SUBDIRS:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return run_dir, int(run_id)

    def ensure_subdirs(self, run_id):
        """Ensure subfolders exist for an existing run id."""
        run_dir, _ = self.create_run_dir(run_id=run_id)
        return run_dir

    def get_run_paths(self, run_id=None, create=False):
        """Return standard artifact paths for one run."""
        if run_id is None:
            run_id = self.get_next_run_id()

        run_dir = self.get_run_dir(run_id)
        if create:
            run_dir, run_id = self.create_run_dir(run_id=run_id)

        return {
            "run_id": int(run_id),
            "run_dir": run_dir,
            "models_dir": run_dir / "models",
            "videos_dir": run_dir / "videos",
            "data_dir": run_dir / "data",
            "tensorboard_dir": run_dir / "tensorboard",
            "monitor_csv": run_dir / "data" / "monitor.csv",
            "thresholds_json": run_dir / "data" / "thresholds.json",
            "params_json": run_dir / "data" / "params.json",
        }

    def save_params(self, run_id, params):
        """Save training params into data/params.json."""
        paths = self.get_run_paths(run_id=run_id, create=True)
        with open(paths["params_json"], "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    def _resolve_monitor_csv(self, run_dir):
        """Support both new and old file layout."""
        new_path = run_dir / "data" / "monitor.csv"
        old_path = run_dir / "monitor.csv"
        if new_path.exists():
            return new_path
        if old_path.exists():
            return old_path
        return None

    def _resolve_thresholds_json(self, run_dir):
        """Support both new and old file layout."""
        new_path = run_dir / "data" / "thresholds.json"
        old_path = run_dir / "thresholds.json"
        if new_path.exists():
            return new_path
        if old_path.exists():
            return old_path
        return None

    def load_run_data(self, run_ids=None):
        """Load monitor.csv and thresholds.json for selected runs."""
        if run_ids is None:
            run_ids = self.list_run_ids()

        data = {}
        for rid in run_ids:
            run_dir = self.get_run_dir(rid)
            if not run_dir.exists():
                continue

            item = {"run_id": int(rid), "run_dir": run_dir}

            csv_path = self._resolve_monitor_csv(run_dir)
            if csv_path is not None:
                try:
                    df = pd.read_csv(csv_path, skiprows=1)
                    if "l" in df.columns and "timesteps" not in df.columns:
                        df["timesteps"] = df["l"].cumsum()
                    item["monitor"] = df
                except Exception as exc:
                    print(f"Error loading monitor file for run {rid}: {exc}")

            json_path = self._resolve_thresholds_json(run_dir)
            if json_path is not None:
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        item["thresholds"] = json.load(f)
                except Exception as exc:
                    print(f"Error loading thresholds file for run {rid}: {exc}")

            data[int(rid)] = item

        return data

    def summarize_results(self, run_ids=None):
        """Return per-threshold mean/min/max/count over selected runs."""
        data = self.load_run_data(run_ids=run_ids)
        by_threshold = {}

        for rid, item in data.items():
            thresholds = item.get("thresholds", {})
            for t, steps in thresholds.items():
                if steps is None:
                    continue
                by_threshold.setdefault(str(t), []).append(int(steps))

        summary = {}
        for t, values in by_threshold.items():
            if not values:
                continue
            summary[t] = {
                "mean": float(sum(values) / len(values)),
                "min": int(min(values)),
                "max": int(max(values)),
                "count": int(len(values)),
            }
        return summary

    def clean_run_dirs(self, run_ids=None):
        """Delete selected run folders or all runs."""
        if run_ids is None:
            run_ids = self.list_run_ids()

        for rid in run_ids:
            run_dir = self.get_run_dir(rid)
            if run_dir.exists():
                shutil.rmtree(run_dir)
                print(f"Cleaned {run_dir}")

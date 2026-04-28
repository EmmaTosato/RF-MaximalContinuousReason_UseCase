#!/usr/bin/env python3
"""Run one baseline dataset through Redis workers and dump Redis afterwards."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiments_utils import is_pid_running, save_readable_dump, wait_for_workers
from helpers import convert_numpy_types
from init_baseline import (
    load_classifier_from_json,
    load_dataset_from_baseline,
    list_available_datasets,
    process_all_classified_samples_baseline,
)
from init_utils import (
    initialize_seed_candidate,
    store_forest_and_endpoints,
    store_training_set,
)
from launch_workers import WorkerManager
from redis_backup import (
    DEFAULT_REDIS_DATABASES,
    create_multi_database_backup,
    save_multi_database_backup_to_directory,
)
from redis_helpers.connection import connect_redis
from redis_helpers.utils import clean_all_databases
from rf_utils import sklearn_forest_to_forest


DEFAULT_OUTPUT_DIR = Path("results") / "dataset_runs"


@dataclass
class InitResult:
    dataset_name: str
    class_label: str
    classifier_path: str
    n_features: int
    train_samples: int
    test_samples: int
    stored_samples: int
    classes: list[str]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def safe_path_component(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "value"


def class_sort_key(value: object) -> tuple[int, float | str]:
    text = str(value)
    try:
        return (0, float(text))
    except (TypeError, ValueError):
        return (1, text)


def parse_database_list(value: str | None) -> list[int]:
    if value is None or value.strip().lower() in {"", "all", "*"}:
        return list(DEFAULT_REDIS_DATABASES)

    databases: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text.strip())
            end = int(end_text.strip())
            if end < start:
                raise ValueError(f"Invalid database range: {part}")
            databases.update(range(start, end + 1))
        else:
            databases.add(int(part))

    if not databases:
        raise ValueError("No Redis databases selected")
    return sorted(databases)


def parse_class_labels(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []

    labels: list[str] = []
    for value in values:
        for part in str(value).split(","):
            label = part.strip()
            if label:
                labels.append(label)
    return labels


def normalize_class_label(requested_label: str, sklearn_classes) -> str:
    forest_labels = [str(label) for label in sklearn_classes]
    if requested_label in forest_labels:
        return requested_label

    for forest_label in forest_labels:
        try:
            if float(forest_label) == float(requested_label):
                print(f"[INFO] Normalizing class label '{requested_label}' to '{forest_label}'")
                return forest_label
        except (TypeError, ValueError):
            continue

    return requested_label


def build_redis_config(args: argparse.Namespace) -> dict[str, object]:
    return {"host": args.redis_host, "port": args.redis_port}


def discover_class_labels(dataset_name: str) -> list[str]:
    _, x_test_samples, _, _, feature_names, _, _ = load_dataset_from_baseline(dataset_name)
    sklearn_rf, _ = load_classifier_from_json(dataset_name)
    forest = sklearn_forest_to_forest(sklearn_rf, feature_names)
    x_test = [dict(zip(feature_names, sample)) for sample in x_test_samples]
    predictions = {forest.predict(sample) for sample in x_test}
    return [str(label) for label in sorted(predictions, key=class_sort_key)]


def initialize_baseline_dataset(
    dataset_name: str,
    class_label: str,
    *,
    redis_host: str,
    redis_port: int,
    clean_redis: bool,
) -> InitResult:
    print(f"\n[START] Initializing dataset '{dataset_name}' for class '{class_label}'")

    connections, db_mapping = connect_redis(host=redis_host, port=redis_port)
    if clean_redis:
        clean_all_databases(connections, db_mapping)

    (
        x_train,
        x_test_samples,
        y_train,
        _,
        feature_names,
        all_classes,
        _,
    ) = load_dataset_from_baseline(dataset_name)

    sklearn_rf, classifier_path = load_classifier_from_json(dataset_name)
    normalized_class_label = normalize_class_label(class_label, sklearn_rf.classes_)
    forest = sklearn_forest_to_forest(sklearn_rf, feature_names)

    store_training_set(
        connections,
        x_train,
        y_train,
        feature_names,
        dataset_name,
        dataset_type="baseline",
    )

    print("\n[INFO] Storing forest and endpoints in Redis")
    eu_data = store_forest_and_endpoints(connections, forest)

    print("\n[INFO] Processing test samples")
    x_test = [dict(zip(feature_names, sample)) for sample in x_test_samples]
    stored_samples, summary = process_all_classified_samples_baseline(
        connections,
        dataset_name,
        normalized_class_label,
        forest,
        x_test,
        eu_data,
    )

    if not stored_samples:
        raise RuntimeError(
            f"No samples classified as '{normalized_class_label}' for dataset '{dataset_name}'"
        )

    print("\n[INFO] Initializing seed candidates")
    for sample in stored_samples:
        meta_json = connections["DATA"].get(f"{sample['sample_key']}_meta")
        if not meta_json:
            print(f"[WARNING] Missing metadata for {sample['sample_key']}")
            continue
        initialize_seed_candidate(connections, json.loads(meta_json), forest, eu_data)

    metadata = {
        "dataset": dataset_name,
        "class_label": normalized_class_label,
        "classifier_path": classifier_path,
        "n_estimators": sklearn_rf.n_estimators,
        "max_depth": sklearn_rf.max_depth,
        "n_features": sklearn_rf.n_features_in_,
        "classes": [str(label) for label in sklearn_rf.classes_],
        "summary": summary,
        "initialized_at_utc": utc_now_iso(),
    }
    connections["DATA"].set("label", normalized_class_label)
    connections["DATA"].set(
        "classifier_metadata",
        json.dumps(convert_numpy_types(metadata), sort_keys=True),
    )

    print(
        f"[SUCCESS] Initialized {len(stored_samples)} seed sample(s) "
        f"for class '{normalized_class_label}'"
    )

    return InitResult(
        dataset_name=dataset_name,
        class_label=normalized_class_label,
        classifier_path=classifier_path,
        n_features=len(feature_names),
        train_samples=len(x_train),
        test_samples=len(x_test_samples),
        stored_samples=len(stored_samples),
        classes=[str(label) for label in all_classes],
    )


def load_running_workers() -> dict[str, dict[str, object]]:
    pids_file = Path("workers") / "worker_pids.json"
    if not pids_file.exists():
        return {}

    try:
        pids_data = json.loads(pids_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    running: dict[str, dict[str, object]] = {}
    for worker_id, info in pids_data.items():
        if not isinstance(info, dict):
            continue
        if is_pid_running(info.get("pid")):
            running[worker_id] = info
    return running


def wait_for_workers_with_timeout(timeout_seconds: float | None) -> bool:
    if timeout_seconds is None:
        wait_for_workers()
        return True

    deadline = time.time() + timeout_seconds
    time.sleep(1)

    while time.time() < deadline:
        running = load_running_workers()
        if not running:
            print("[EXPERIMENT] All workers exited.")
            return True
        time.sleep(2)

    print(f"[WARNING] Run timeout reached after {timeout_seconds:.2f} seconds")
    return False


def archive_new_logs(logs_before: set[Path], destination: Path) -> list[str]:
    log_dir = Path("logs")
    logs_after = set(log_dir.glob("*.log")) if log_dir.exists() else set()
    new_logs = sorted(logs_after - logs_before)
    if not new_logs:
        return []

    destination.mkdir(parents=True, exist_ok=True)
    archived: list[str] = []
    for log_file in new_logs:
        target = destination / log_file.name
        try:
            shutil.copy2(log_file, target)
            archived.append(str(target))
        except OSError as exc:
            print(f"[WARNING] Failed to archive log {log_file}: {exc}")
    return archived


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(convert_numpy_types(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_class(
    args: argparse.Namespace,
    run_root: Path,
    requested_class_label: str,
    databases: list[int],
) -> dict[str, object]:
    class_started = time.time()
    class_dir = run_root / f"class_{safe_path_component(requested_class_label)}"
    class_dir.mkdir(parents=True, exist_ok=True)

    init_result = initialize_baseline_dataset(
        args.dataset_name,
        requested_class_label,
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        clean_redis=not args.no_clean,
    )

    redis_config = build_redis_config(args)
    manager = WorkerManager(args.worker_config)
    manager.config["redis"] = redis_config

    existing_workers = load_running_workers()
    if existing_workers:
        if not args.stop_existing_workers:
            worker_list = ", ".join(sorted(existing_workers))
            raise RuntimeError(
                "Workers are already running from workers/worker_pids.json: "
                f"{worker_list}. Stop them first or use --stop-existing-workers."
            )
        manager.stop_workers(list(existing_workers.keys()))

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logs_before = set(log_dir.glob("*.log"))

    print(f"\n[EXPERIMENT] Launching {args.workers} worker(s)")
    started = manager.start_workers(
        args.workers,
        worker_script=args.worker_script,
        one_solution=args.one_solution,
        sample_timeout=args.sample_timeout,
        use_R_cache=args.use_R_cache,
        use_GP_cache=args.use_GP_cache,
        use_NR_cache=args.use_NR_cache,
        use_BP_cache=args.use_BP_cache,
    )
    if not started:
        raise RuntimeError("Worker launch failed")

    completed = wait_for_workers_with_timeout(args.run_timeout)
    if not completed:
        manager.stop_workers()

    archived_logs = archive_new_logs(logs_before, class_dir / "logs")

    print(f"\n[DUMP] Exporting Redis DBs {databases} to {class_dir}")
    backups = create_multi_database_backup(
        redis_config,
        databases=databases,
        scan_count=args.scan_count,
    )
    dump_paths = save_multi_database_backup_to_directory(
        backups,
        class_dir,
        file_prefix=args.dump_prefix,
    )

    if args.readable_dump:
        save_readable_dump(backups, class_dir, redis_config=redis_config)

    duration = time.time() - class_started
    metadata = {
        "dataset": args.dataset_name,
        "requested_class_label": requested_class_label,
        "class_label": init_result.class_label,
        "workers": args.workers,
        "worker_script": args.worker_script,
        "sample_timeout": args.sample_timeout,
        "run_timeout": args.run_timeout,
        "completed": completed,
        "duration_seconds": duration,
        "redis": redis_config,
        "databases": databases,
        "dump_key_counts": {
            str(db): payload.get("metadata", {}).get("key_count", 0)
            for db, payload in backups.items()
        },
        "dump_files": {str(db): str(path) for db, path in dump_paths.items()},
        "readable_dump": str(class_dir / "redis_dump_readable.json")
        if args.readable_dump
        else None,
        "archived_logs": archived_logs,
        "init": init_result.__dict__,
        "finished_at_utc": utc_now_iso(),
    }
    write_json(class_dir / "run_metadata.json", metadata)

    print(
        f"[SUCCESS] Class '{init_result.class_label}' finished in "
        f"{duration:.2f}s; dump saved in {class_dir}"
    )
    return metadata


def make_run_root(output_dir: Path, dataset_name: str, run_id: str | None) -> Path:
    if run_id is None:
        run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_dir / safe_path_component(dataset_name) / safe_path_component(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize a baseline dataset, run workers, and dump Redis DBs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_name", nargs="?", help="Baseline dataset name")
    parser.add_argument(
        "--class-label",
        action="append",
        dest="class_labels",
        help="Class label to run. Can be repeated or comma-separated. Defaults to all predicted classes.",
    )
    parser.add_argument("--list-datasets", action="store_true", help="List available baseline datasets")
    parser.add_argument("--workers", "--num-workers", type=int, default=4, help="Worker count")
    parser.add_argument("--worker-script", default="worker_cache_logged.py", help="Worker script path")
    parser.add_argument("--worker-config", default=None, help="Optional launch_workers.py YAML config")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument(
        "--databases",
        default="0-10",
        help="Redis DBs to dump, e.g. '0-10', '0,2,8', or 'all'",
    )
    parser.add_argument("--scan-count", type=int, default=1000, help="Redis SCAN count hint")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--run-id", default=None, help="Optional run directory name")
    parser.add_argument("--dump-prefix", default="redis_backup", help="Dump file prefix")
    parser.add_argument("--no-readable-dump", dest="readable_dump", action="store_false", help="Skip readable dump")
    parser.set_defaults(readable_dump=True)
    parser.add_argument("--no-clean", action="store_true", help="Do not flush Redis before initialization")
    parser.add_argument(
        "--stop-existing-workers",
        action="store_true",
        help="Stop workers listed in workers/worker_pids.json before starting this run",
    )
    parser.add_argument("--one-solution", action="store_true", help="Pass --one-solution to workers")
    parser.add_argument("--sample-timeout", type=float, default=None, help="Per-sample timeout passed to workers")
    parser.add_argument("--run-timeout", type=float, default=None, help="Stop workers after this many seconds")
    parser.add_argument("--use-R-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-GP-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-NR-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-BP-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_datasets:
        list_available_datasets()
        return 0

    if not args.dataset_name:
        parser.error("dataset_name is required unless --list-datasets is used")

    try:
        databases = parse_database_list(args.databases)
        class_labels = parse_class_labels(args.class_labels)
        if not class_labels:
            print(f"[INFO] Discovering predicted classes for '{args.dataset_name}'")
            class_labels = discover_class_labels(args.dataset_name)
            print(f"[INFO] Classes to run: {', '.join(class_labels)}")

        run_root = make_run_root(args.output_dir, args.dataset_name, args.run_id)
        run_started = time.time()
        run_metadata: list[dict[str, object]] = []

        for class_label in class_labels:
            run_metadata.append(run_class(args, run_root, class_label, databases))

        summary = {
            "dataset": args.dataset_name,
            "classes": class_labels,
            "run_root": str(run_root),
            "duration_seconds": time.time() - run_started,
            "finished_at_utc": utc_now_iso(),
            "class_runs": run_metadata,
        }
        write_json(run_root / "dataset_run_metadata.json", summary)
        print(f"\n[DONE] Dataset run complete. Outputs: {run_root}")
        return 0

    except KeyboardInterrupt:
        print("\n[ABORT] Interrupted by user")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

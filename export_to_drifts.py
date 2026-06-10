"""Export run.py model artefacts to the MCR directory structure for init_dsd.py.

Reads files saved by run.py under <run_dir>/model/ and copies them into the
RF-MaximalContinuousReason_UseCase/baseline/resources/datasets/ layout that
init_dsd.py expects.  Also prints the ready-to-run init_dsd.py command.

Dataset name format:
    {experiment}_fs{n}_fold{k}
    e.g. amyloid_pos_vs_neg_fs3_fold1

Feature set numbering:
    base            → fs1
    base_bio        → fs2
    base_cogn_cli   → fs3
    base_cog_clin_bio → fs4

Usage:
    conda activate emma
    python -m src.pipeline.export_to_drifts \\
      --run-dir results_classification/exp2/amyloid_pos_vs_neg/base_cogn_cli/kf5_tune \\
      --fold 1 \\
      --class-label 1

    # Export all folds at once
    python -m src.pipeline.export_to_drifts \\
      --run-dir results_classification/exp2/amyloid_pos_vs_neg/base_cogn_cli/kf5_tune \\
      --class-label 1
"""
import argparse
import json
import os
import shutil
from pathlib import Path


_MCR_DIR_NAME = "RF-MaximalContinuousReason_UseCase"

_FS_NUMBER: dict[str, int] = {
    "base": 1,
    "base_bio": 2,
    "base_cogn_cli": 3,
    "base_cog_clin_bio": 4,
}


def _build_dataset_name(experiment: str, feature_type: str, fold: int) -> str:
    """Build the MCR dataset name: {experiment}_fs{n}_fold{k}.

    Raises
    ------
    ValueError
        If feature_type has no registered number in _FS_NUMBER.
    """
    if feature_type not in _FS_NUMBER:
        raise ValueError(
            f"Unknown feature_type '{feature_type}'. "
            f"Add it to _FS_NUMBER in export_to_drifts.py. "
            f"Known: {list(_FS_NUMBER)}"
        )
    return f"{experiment}_fs{_FS_NUMBER[feature_type]}_fold{fold}"


def export_fold(
    run_dir: Path,
    fold: int,
    mcr_dir: Path,
    experiment_name: str,
    feature_type: str,
) -> tuple[Path, Path]:
    """Copy CSV and .samples for one fold into the MCR dataset directory.

    Returns
    -------
    dataset_dir : Path
        The MCR dataset directory created.
    pkl_path : Path
        Absolute path to the .pkl model (for --model-file).
    """
    model_dir = run_dir / "model"
    dataset_name = _build_dataset_name(experiment_name, feature_type, fold)
    dataset_dir = mcr_dir / "baseline" / "resources" / "datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # CSV: all subjects — copied and renamed to <dataset_name>.csv
    csv_src = model_dir / f"{experiment_name}.csv"
    if not csv_src.exists():
        raise FileNotFoundError(f"CSV not found: {csv_src}")
    shutil.copy2(csv_src, dataset_dir / f"{dataset_name}.csv")

    # .samples: fold-specific — renamed to <dataset_name>.samples
    samples_src = model_dir / f"{experiment_name}_fold{fold}.samples"
    if not samples_src.exists():
        raise FileNotFoundError(f"Samples not found: {samples_src}")
    shutil.copy2(samples_src, dataset_dir / f"{dataset_name}.samples")

    # _meta.json: fold-specific — renamed to <dataset_name>_meta.json
    meta_src = model_dir / f"{experiment_name}_fold{fold}_meta.json"
    if not meta_src.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_src}")
    shutil.copy2(meta_src, dataset_dir / f"{dataset_name}_meta.json")

    # .pkl: not copied, passed explicitly via --model-file
    pkl_path = (model_dir / f"{experiment_name}_fold{fold}.pkl").resolve()
    if not pkl_path.exists():
        raise FileNotFoundError(f"Model pickle not found: {pkl_path}")

    return dataset_dir, pkl_path


def _print_command(
    mcr_dir: Path,
    dataset_name: str,
    pkl_path: Path,
    class_label: str,
    redis_port: int,
) -> None:
    model_file_arg = os.path.relpath(pkl_path, mcr_dir)
    print(
        f"\n  cd {mcr_dir}\n"
        f"  python init_dsd.py {dataset_name} \\\n"
        f"      --model-file {model_file_arg} \\\n"
        f"      --class-label {class_label} \\\n"
        f"      --redis-port {redis_port}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export run.py artefacts to MCR directory for init_dsd.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help=(
            "Path to run output directory "
            "(e.g. results_classification/exp2/amyloid_pos_vs_neg/base_cogn_cli/kf5_tune)"
        ),
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Fold number to export (default: all folds found in model/)",
    )
    parser.add_argument(
        "--class-label", required=True, type=str,
        help="Class label to pass to init_dsd.py (e.g. 1 or 'Positive')",
    )
    parser.add_argument(
        "--redis-port", type=int, default=6379,
        help="Redis port for the printed init_dsd.py command (default: 6379)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    run_dir = (
        (project_root / args.run_dir).resolve()
        if not args.run_dir.is_absolute()
        else args.run_dir.resolve()
    )
    mcr_dir = project_root / _MCR_DIR_NAME

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    if not mcr_dir.exists():
        raise FileNotFoundError(f"MCR submodule not found: {mcr_dir}")

    snapshot_path = run_dir / "config_snapshot.json"
    if not snapshot_path.exists():
        raise FileNotFoundError(f"config_snapshot.json not found in: {run_dir}")
    try:
        with open(snapshot_path) as f:
            snapshot = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse config_snapshot.json at {snapshot_path}: {e}"
        ) from e

    experiment_name: str = snapshot["experiment"]
    feature_type: str = snapshot["config"]["feature_type"]
    if feature_type not in _FS_NUMBER:
        raise ValueError(
            f"Unknown feature_type '{feature_type}' in config_snapshot.json. "
            f"Add it to _FS_NUMBER in export_to_drifts.py. "
            f"Known: {list(_FS_NUMBER)}"
        )

    model_dir = run_dir / "model"
    if args.fold is not None:
        folds = [args.fold]
    else:
        folds = sorted(
            int(p.stem.split("_fold")[-1])
            for p in model_dir.glob(f"{experiment_name}_fold*.pkl")
        )
        if not folds:
            raise FileNotFoundError(
                f"No .pkl files found in {model_dir} matching "
                f"'{experiment_name}_fold*.pkl'"
            )

    print(f"Exporting {len(folds)} fold(s)")
    print(f"  experiment   : {experiment_name}")
    print(f"  feature_type : {feature_type} (fs{_FS_NUMBER[feature_type]})")
    print(f"  run_dir      : {run_dir}")
    print(f"  mcr_dir      : {mcr_dir}")

    for fold in folds:
        dataset_name = _build_dataset_name(experiment_name, feature_type, fold)
        dataset_dir, pkl_path = export_fold(
            run_dir, fold, mcr_dir, experiment_name, feature_type
        )
        print(f"\n[fold {fold}] dataset_name : {dataset_name}")
        print(f"            dataset_dir  : {dataset_dir}")
        print(f"  Command to run:")
        _print_command(mcr_dir, dataset_name, pkl_path, args.class_label, args.redis_port)


if __name__ == "__main__":
    main()

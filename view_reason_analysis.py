#!/usr/bin/env python3
"""Build an HTML report with the same visualizations used by reasons_analysis.ipynb."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any


REASON_TYPES = ("reasons", "non_reasons", "anti_reasons")


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_dir:
        checkpoint = Path(args.checkpoint_dir)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint}")
        return checkpoint

    from etl.loader import list_checkpoints

    checkpoints = list_checkpoints(Path(args.results_dir))
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found under {args.results_dir}. "
            "Pass --checkpoint-dir explicitly."
        )
    if args.checkpoint_index < 0 or args.checkpoint_index >= len(checkpoints):
        raise IndexError(
            f"--checkpoint-index {args.checkpoint_index} is out of range "
            f"for {len(checkpoints)} checkpoint(s)."
        )
    return checkpoints[args.checkpoint_index]


def infer_dataset_name_from_path(checkpoint_dir: Path) -> str | None:
    parts = checkpoint_dir.parts
    for marker in ("checkpoints", "dataset_runs", "res_ablation"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def unwrap_value(value: Any) -> Any:
    if isinstance(value, dict) and "value_json" in value:
        return value["value_json"]
    return value


def get_eu(db: dict[str, Any]) -> dict[str, Any]:
    return unwrap_value(db["data"]["EU"])


def get_training_set(db: dict[str, Any]) -> dict[str, Any]:
    return unwrap_value(db["data"]["TRAINING_SET"])


def calculate_cost_dataframe(
    db: dict[str, Any],
    tests_sample: dict[str, Any],
    test_ids: list[str],
    sigmas_all: dict[str, Any],
    *,
    max_bitmaps_per_type: int | None = None,
):
    import pandas as pd
    from cost_function import cost_function
    from redis_helpers.icf import bitmap_to_icf

    eu = get_eu(db)
    expected_bitmap_len = sum(len(values) - 1 for values in eu.values()) + len(eu)
    records: list[dict[str, Any]] = []

    print(f"Expected bitmap length: {expected_bitmap_len}")
    for reason_type in REASON_TYPES:
        db_bucket = db.get(reason_type) or {}
        bitmaps = [
            key
            for key in db_bucket.keys()
            if isinstance(key, str)
            and key
            and len(key) == expected_bitmap_len
            and set(key) <= {"0", "1"}
        ]
        if max_bitmaps_per_type is not None:
            bitmaps = bitmaps[:max_bitmaps_per_type]

        print(f"  {reason_type}: {len(bitmaps)} bitmap(s)")
        for bitmap_string in bitmaps:
            try:
                icf = bitmap_to_icf(bitmap_string, eu)
            except Exception as exc:
                print(f"    skip bitmap len={len(bitmap_string)}: {exc}")
                continue

            for sample_id in test_ids:
                sample_data = tests_sample[sample_id]
                sample_data.setdefault(reason_type, {})
                cost = cost_function(
                    sample=sample_data["features"],
                    sigmas=sigmas_all[sample_id],
                    icf=icf,
                )
                sample_data[reason_type][bitmap_string] = {"icf": icf, "cost": cost}
                records.append(
                    {
                        "reason_type": reason_type,
                        "bitmap": bitmap_string,
                        "sample_id": sample_id,
                        "cost": cost,
                        "icf": icf,
                    }
                )

    return pd.DataFrame(records)


def choose_sample_id(
    requested_sample_id: str | None,
    test_ids: list[str],
    tests_sample: dict[str, Any],
) -> str:
    if requested_sample_id:
        if requested_sample_id not in tests_sample:
            raise KeyError(f"Sample id not found: {requested_sample_id}")
        return requested_sample_id

    for sample_id in test_ids:
        sample_data = tests_sample[sample_id]
        if sample_data.get("reasons") and sample_data.get("anti_reasons"):
            return sample_id
    return test_ids[0]


def figure_to_html(fig: Any, *, include_plotlyjs: bool) -> str:
    if fig is None:
        return ""
    try:
        return fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    except ModuleNotFoundError as exc:
        if exc.name == "plotly":
            raise RuntimeError(
                "Plotly is required for the visual report. Install it with `pip install plotly`."
            ) from exc
        raise


def make_table_html(df: pd.DataFrame, columns: list[str], max_rows: int = 15) -> str:
    if df.empty:
        return "<p>No rows available.</p>"
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return "<p>No requested columns available.</p>"
    return df[existing].head(max_rows).to_html(index=False, border=0, classes="data-table")


def write_report(
    output_path: Path,
    *,
    dataset_name: str,
    checkpoint_dir: Path,
    sample_id: str,
    tests_sample: dict[str, Any],
    test_ids: list[str],
    feature_names: list[str],
    cost_df: pd.DataFrame,
    sample_robustness_df: pd.DataFrame,
    bitmap_robustness: pd.DataFrame,
    figures: list[tuple[str, Any]],
) -> None:
    from helpers import convert_numpy_types

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_sections = []
    include_plotlyjs = True
    for title, fig in figures:
        if fig is None:
            continue
        figure_sections.append(
            f"<section><h2>{html.escape(title)}</h2>"
            f"{figure_to_html(fig, include_plotlyjs=include_plotlyjs)}</section>"
        )
        include_plotlyjs = False

    sample_meta = tests_sample.get(sample_id, {})
    counts = (
        cost_df.groupby("reason_type")["bitmap"].nunique().to_dict()
        if not cost_df.empty
        else {}
    )
    robustness_values = sample_robustness_df["robustness"].dropna()
    mean_robustness = robustness_values.mean() if len(robustness_values) else None

    summary = {
        "dataset": dataset_name,
        "checkpoint_dir": str(checkpoint_dir),
        "sample_id": sample_id,
        "features": len(feature_names),
        "test_samples": len(test_ids),
        "reason_counts": counts,
        "mean_sample_robustness": mean_robustness,
        "sample_predicted_label": sample_meta.get("predicted_label"),
        "sample_actual_label": sample_meta.get("actual_label"),
        "sample_prediction_correct": sample_meta.get("prediction_correct"),
    }

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Reasons Analysis - {html.escape(dataset_name)}</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #1f2933;
      background: #f6f8fb;
    }}
    header {{
      padding: 28px 36px;
      background: #ffffff;
      border-bottom: 1px solid #dde3ea;
    }}
    main {{
      max-width: 1220px;
      margin: 0 auto;
      padding: 24px 28px 44px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 20px; margin-top: 22px; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .metric {{
      background: #ffffff;
      border: 1px solid #dde3ea;
      border-radius: 6px;
      padding: 12px 14px;
    }}
    .metric strong {{
      display: block;
      font-size: 12px;
      color: #52616f;
      margin-bottom: 4px;
      text-transform: uppercase;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #dde3ea;
      border-radius: 6px;
      padding: 18px;
      margin-top: 18px;
    }}
    pre {{
      white-space: pre-wrap;
      background: #101820;
      color: #e6edf3;
      padding: 14px;
      border-radius: 6px;
      overflow-x: auto;
    }}
    .data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid #e5eaf0;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{
      color: #52616f;
      background: #f6f8fb;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Reasons Analysis - {html.escape(dataset_name)}</h1>
    <div>{html.escape(str(checkpoint_dir))}</div>
  </header>
  <main>
    <section>
      <h2>Summary</h2>
      <div class="meta-grid">
        <div class="metric"><strong>Sample</strong>{html.escape(sample_id)}</div>
        <div class="metric"><strong>Features</strong>{len(feature_names)}</div>
        <div class="metric"><strong>Test Samples</strong>{len(test_ids)}</div>
        <div class="metric"><strong>Cost Rows</strong>{len(cost_df)}</div>
        <div class="metric"><strong>Mean Robustness</strong>{'' if mean_robustness is None else f'{mean_robustness:.6f}'}</div>
      </div>
      <pre>{html.escape(json.dumps(convert_numpy_types(summary), indent=2, sort_keys=True))}</pre>
    </section>
    <section>
      <h2>Anti-Reason Robustness Table</h2>
      {make_table_html(bitmap_robustness, ['max_cost', 'robustness', 'mean_cost', 'n_samples'])}
    </section>
    <section>
      <h2>Sample Robustness Table</h2>
      {make_table_html(sample_robustness_df, ['sample_id', 'robustness', 'predicted_label', 'actual_label', 'correct_prediction'])}
    </section>
    {''.join(figure_sections)}
  </main>
</body>
</html>
"""
    output_path.write_text(html_doc, encoding="utf-8")


def write_outputs(
    output_dir: Path,
    *,
    cost_df: pd.DataFrame,
    sample_robustness_df: pd.DataFrame,
    bitmap_robustness: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cost_df.empty:
        cost_df.drop(columns=["icf"], errors="ignore").to_csv(output_dir / "reason_costs.csv", index=False)
    sample_robustness_df.to_csv(output_dir / "sample_robustness.csv", index=False)
    bitmap_robustness.to_csv(output_dir / "anti_reasons_robustness.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reasons_analysis-style HTML visualizations from a Redis checkpoint dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", type=Path, help="Directory containing redis_backup_db*.json")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory to scan for checkpoints")
    parser.add_argument("--checkpoint-index", type=int, default=0, help="Checkpoint index when scanning --results-dir")
    parser.add_argument("--dataset-name", help="Dataset name shown in the report")
    parser.add_argument("--sample-id", help="Sample id to visualize. Defaults to first sample with reason and anti-reason.")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "reason_visualizations")
    parser.add_argument("--output-name", default="reason_analysis_report.html", help="Report HTML filename")
    parser.add_argument("--max-bitmaps-per-type", type=int, default=None, help="Limit bitmaps per reason type")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        checkpoint_dir = resolve_checkpoint(args)
        from cost_function import cal_sigmas
        from etl.loader import etl_from_dir
        from etl.reasons_analysis import (
            calculate_all_samples_robustness,
            calculate_robustness_per_bitmap,
            create_robustness_visualizations,
            extract_test_samples,
            visualize_anti_reason_corridor,
            visualize_sample_comparison,
            visualize_sample_comparison_smooth,
            visualize_sample_with_icf,
        )

        inferred_dataset_name = args.dataset_name or infer_dataset_name_from_path(checkpoint_dir)
        db = etl_from_dir(checkpoint_dir, dataset_name=inferred_dataset_name, verbose=True)
        dataset_name = db.get("_dataset_name", checkpoint_dir.name)

        tests_sample, x_test, test_ids, feature_names = extract_test_samples(db)
        if not test_ids:
            raise RuntimeError("No test samples found in checkpoint DATA database")

        training_set = get_training_set(db)
        sigmas_all = cal_sigmas(
            training_set["X_train"],
            x_test,
            feature_names,
            test_ids=test_ids,
        )

        print("\nCalculating costs...")
        cost_df = calculate_cost_dataframe(
            db,
            tests_sample,
            test_ids,
            sigmas_all,
            max_bitmaps_per_type=args.max_bitmaps_per_type,
        )
        if cost_df.empty:
            raise RuntimeError("No reason costs could be calculated from this checkpoint")

        num_features = len(feature_names)
        anti_reasons_df = cost_df[cost_df["reason_type"] == "anti_reasons"].copy()
        if anti_reasons_df.empty:
            raise RuntimeError("No anti_reasons found; robustness visualizations need DB5/AR data")

        bitmap_robustness = calculate_robustness_per_bitmap(
            anti_reasons_df,
            num_features=num_features,
        )
        sample_robustness_df = calculate_all_samples_robustness(
            cost_df=cost_df,
            num_features=num_features,
            tests_sample=tests_sample,
            test_ids=test_ids,
            verbose=True,
        )
        sample_id = choose_sample_id(args.sample_id, test_ids, tests_sample)
        print(f"\nVisualizing sample: {sample_id}")

        fig_main, fig_quartiles = create_robustness_visualizations(
            sample_robustness_df=sample_robustness_df,
            dataset_name=str(dataset_name),
        )
        figures = [
            ("Sample-Level Robustness", fig_main),
            ("Robustness Quartiles", fig_quartiles),
            ("Sample With Maximal Reason", visualize_sample_with_icf(sample_id, tests_sample, feature_names, reason_type="reasons")),
            ("Sample With Anti-Reason", visualize_sample_with_icf(sample_id, tests_sample, feature_names, reason_type="anti_reasons")),
            ("Reason vs Anti-Reason", visualize_sample_comparison(sample_id, tests_sample, feature_names)),
            ("Smooth Corridor Comparison", visualize_sample_comparison_smooth(sample_id, tests_sample, feature_names)),
            ("Anti-Reason Corridor", visualize_anti_reason_corridor(sample_id, tests_sample, feature_names)),
        ]

        output_dir = args.output_dir / str(dataset_name) / checkpoint_dir.name
        write_outputs(
            output_dir,
            cost_df=cost_df,
            sample_robustness_df=sample_robustness_df,
            bitmap_robustness=bitmap_robustness,
        )
        report_path = output_dir / args.output_name
        write_report(
            report_path,
            dataset_name=str(dataset_name),
            checkpoint_dir=checkpoint_dir,
            sample_id=sample_id,
            tests_sample=tests_sample,
            test_ids=test_ids,
            feature_names=feature_names,
            cost_df=cost_df,
            sample_robustness_df=sample_robustness_df,
            bitmap_robustness=bitmap_robustness,
            figures=figures,
        )

        print(f"\nReport written to: {report_path}")
        print(f"CSV outputs written to: {output_dir}")
        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except ModuleNotFoundError as exc:
        if exc.name == "plotly":
            print(
                "\nERROR: Plotly is required to generate the HTML report. "
                "Install it with `pip install plotly`.",
                file=sys.stderr,
            )
            return 1
        raise
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

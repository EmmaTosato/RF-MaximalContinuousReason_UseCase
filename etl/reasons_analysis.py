"""
ETL functions for reasons analysis - data extraction and processing
"""
from typing import Dict, List, Tuple, Any


def _format_feature_label(feature_name: Any, fallback_index: int) -> str:
    """
    Convert feature identifiers like 't_007' into plain numeric labels.
    Falls back to the original string or positional index if parsing fails.
    """
    if feature_name is None:
        return str(fallback_index)

    feature_str = str(feature_name)
    digits = ''.join(ch for ch in feature_str if ch.isdigit())
    if digits:
        try:
            return str(int(digits))
        except ValueError:
            pass

    return feature_str if feature_str else str(fallback_index)


def extract_test_samples(db: Dict[str, Any]) -> Tuple[Dict, List, List, List]:
    """
    Extract test samples from database

    Parameters
    ----------
    db : dict
        Database dictionary

    Returns
    -------
    tuple
        (tests_sample, X_test, test_ids, feature_names)
    """
    training_set_raw = db["data"]["TRAINING_SET"]
    # Support both flat structure (from etl_from_dir) and value_json-wrapped structure
    training_set = training_set_raw.get("value_json", training_set_raw)
    feature_names = training_set["feature_names"]

    tests_sample = {}
    X_test = []
    test_ids = []

    prefix = "sample_"
    suffix = "_meta"

    for raw_k, v in db["data"].items():
        if isinstance(raw_k, str) and raw_k.startswith(prefix) and raw_k.endswith(suffix):
            sample_id = raw_k[len(prefix):-len(suffix)]
            # Support both flat and value_json-wrapped structures
            meta = v.get("value_json", v) if isinstance(v, dict) else v

            sample_dict = meta.get("sample_dict")
            if sample_dict is None:
                print(f"[WARNING] sample_dict missing for {sample_id}, skipping")
                continue

            tests_sample[sample_id] = {
                **meta,
                "features": sample_dict
            }

            X_test.append(sample_dict)
            test_ids.append(sample_id)

    return tests_sample, X_test, test_ids, feature_names


def calculate_costs_for_reasons(db: Dict, tests_sample: Dict, sigmas_all: Dict) -> Tuple[Dict, List]:
    """
    Calculate costs for all reason types

    Parameters
    ----------
    db : dict
        Database with reasons, non_reasons, anti_reasons
    tests_sample : dict
        Test samples dictionary
    sigmas_all : dict
        Sigma values for all samples

    Returns
    -------
    tuple
        (robustness, cost_records)
    """
    from redis_helpers.icf import bitmap_to_icf
    from cost_function import cost_function

    eu_raw = db["data"]['EU']
    eu = eu_raw.get("value_json", eu_raw) if isinstance(eu_raw, dict) and "value_json" in eu_raw else eu_raw

    robustness = {}
    cost_records = []
    reason_types = ["reasons", "non_reasons", "anti_reasons"]

    for r in reason_types:
        robustness[r] = {"cost": None, "icf": None, "bitmap": None}
        for bitmap_string in db[r].keys():
            icf = bitmap_to_icf(bitmap_string, eu)

            bitmap_max_cost = None
            for sample_id, sample_data in tests_sample.items():
                sample_data.setdefault(r, {})
                sample_entry = sample_data[r].setdefault(bitmap_string, {})

                sample_entry["icf"] = icf
                cost = cost_function(
                    sample=sample_data["features"],
                    sigmas=sigmas_all[sample_id],
                    icf=icf
                )
                sample_entry["cost"] = cost

                cost_records.append({
                    "reason_type": r,
                    "bitmap": bitmap_string,
                    "sample_id": sample_id,
                    "cost": cost,
                })

                if bitmap_max_cost is None or cost > bitmap_max_cost:
                    bitmap_max_cost = cost

            # Update robustness summary only if this bitmap has the highest cost seen so far for this reason type
            if bitmap_max_cost is not None and (robustness[r]["cost"] is None or bitmap_max_cost > robustness[r]["cost"]):
                robustness[r]["cost"] = bitmap_max_cost
                robustness[r]["icf"] = icf
                robustness[r]["bitmap"] = bitmap_string

    return robustness, cost_records


def prepare_cost_dataframe(tests_sample: Dict) -> 'pd.DataFrame':
    """
    Prepare cost DataFrame from tests_sample structure

    Parameters
    ----------
    tests_sample : dict
        Test samples with cost data

    Returns
    -------
    pd.DataFrame
        Cost data in tabular format
    """
    import pandas as pd

    cost_data = []
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        for sample_id, sample_data in tests_sample.items():
            if reason_type not in sample_data:
                continue

            for bitmap_string, reason_data in sample_data[reason_type].items():
                cost_data.append({
                    'sample_id': sample_id,
                    'bitmap_index': bitmap_string,
                    'reason_type': reason_type,
                    'cost': reason_data['cost'],
                    'icf': reason_data.get('icf', {})
                })

    return pd.DataFrame(cost_data)


def calculate_robustness_per_bitmap(cost_df: 'pd.DataFrame', num_features: int = None) -> 'pd.DataFrame':
    """
    Calculate robustness for each bitmap according to IEEE CAI 2026 paper definition.

    Formula from paper Definition (Robustness):
        r(C, x) = 1 - max_{ICF in AR_{C,y}} cost_x(ICF) / |features|

    Where:
        - C is the classifier
        - x is the sample
        - y = C(x) is the classification
        - AR_{C,y} is the set of Anti Reasons for class y
        - cost_x(ICF) is the cost function for ICF applied to sample x
        - |features| is the total number of features

    For each bitmap (which represents an ICF), we calculate:
        - max_cost: max{cost(s, bitmap) : s in test_set}
        - robustness: 1 - max_cost / |features| (if num_features provided)

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with columns: reason_type, bitmap_index, sample_id, cost
    num_features : int, optional
        Number of features for normalization. If None, robustness equals max_cost

    Returns
    -------
    pd.DataFrame
        Robustness statistics per bitmap with columns:
        reason_type, bitmap_index, max_cost, robustness, mean_cost, min_cost,
        std_cost, n_samples
    """
    import pandas as pd

    # Support both 'bitmap' and 'bitmap_index' column names
    bitmap_col = 'bitmap_index' if 'bitmap_index' in cost_df.columns else 'bitmap'

    # Group by bitmap and calculate statistics
    bitmap_robustness = cost_df.groupby(['reason_type', bitmap_col]).agg({
        'cost': ['max', 'mean', 'min', 'std', 'count']
    }).reset_index()

    # Rename columns
    bitmap_robustness.columns = [
        'reason_type', 'bitmap_index', 'max_cost',
        'mean_cost', 'min_cost', 'std_cost', 'n_samples'
    ]

    # Calculate robustness according to paper: r(C,x) = 1 - max_cost / |features|
    if num_features is not None and num_features > 0:
        bitmap_robustness['robustness'] = 1.0 - (bitmap_robustness['max_cost'] / num_features)
    else:
        # If num_features not provided, robustness is just the max_cost
        bitmap_robustness['robustness'] = bitmap_robustness['max_cost']

    # Sort by max_cost descending (higher cost = lower robustness when normalized)
    bitmap_robustness = bitmap_robustness.sort_values('max_cost', ascending=False)

    return bitmap_robustness


def calculate_robustness_summary(robustness_dict: Dict) -> Dict[str, float]:
    """
    Extract robustness summary from the robustness dictionary.

    Parameters
    ----------
    robustness_dict : dict
        Dictionary with keys 'reasons', 'non_reasons', 'anti_reasons',
        each containing 'cost', 'sample', 'icf', 'bitmap'

    Returns
    -------
    dict
        Summary with reason_type -> robustness value
    """
    summary = {}
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        if reason_type in robustness_dict and 'cost' in robustness_dict[reason_type]:
            summary[reason_type] = robustness_dict[reason_type]['cost']

    return summary


def get_most_robust_bitmaps(bitmap_robustness_df: 'pd.DataFrame', top_n: int = 10) -> 'pd.DataFrame':
    """
    Get the top N most robust bitmaps.

    Parameters
    ----------
    bitmap_robustness_df : pd.DataFrame
        DataFrame from calculate_robustness_per_bitmap
    top_n : int
        Number of top bitmaps to return

    Returns
    -------
    pd.DataFrame
        Top N most robust bitmaps
    """
    return bitmap_robustness_df.head(top_n)


def get_robustness_statistics_by_type(bitmap_robustness_df: 'pd.DataFrame') -> Dict[str, Dict[str, float]]:
    """
    Calculate robustness statistics for each reason type.

    Parameters
    ----------
    bitmap_robustness_df : pd.DataFrame
        DataFrame from calculate_robustness_per_bitmap

    Returns
    -------
    dict
        Nested dictionary with reason_type -> {statistic -> value}
    """
    import pandas as pd

    stats = {}
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        subset = bitmap_robustness_df[bitmap_robustness_df['reason_type'] == reason_type]
        if len(subset) > 0:
            stats[reason_type] = {
                'count': len(subset),
                'max_cost': float(subset['max_cost'].max()),
                'mean_max_cost': float(subset['max_cost'].mean()),
                'min_max_cost': float(subset['max_cost'].min()),
                'max_robustness': float(subset['robustness'].max()) if 'robustness' in subset.columns else None,
                'mean_robustness': float(subset['robustness'].mean()) if 'robustness' in subset.columns else None,
                'min_robustness': float(subset['robustness'].min()) if 'robustness' in subset.columns else None,
                'std_robustness': float(subset['robustness'].std()) if 'robustness' in subset.columns else None,
                'median_robustness': float(subset['robustness'].median()) if 'robustness' in subset.columns else None
            }

    return stats


def calculate_sample_robustness(cost_df: 'pd.DataFrame', num_features: int, sample_id: str = None) -> Dict[str, float]:
    """
    Calculate robustness for a specific sample according to IEEE CAI 2026 paper.

    Formula from paper Definition (Robustness):
        r(C, x) = 1 - max_{ICF in AR_{C,y}} cost_x(ICF) / |features|

    Where AR_{C,y} is the set of Anti Reasons for the sample's classification y.

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with costs for the sample(s)
    num_features : int
        Number of features for normalization (|features| in the formula)
    sample_id : str, optional
        Sample ID to filter. If None, uses all data

    Returns
    -------
    dict
        Dictionary with robustness values for each reason type.
        Keys: 'reasons', 'non_reasons', 'anti_reasons'
        Values: robustness score in [0, 1] or None if no data
    """
    if sample_id is not None:
        sample_costs = cost_df[cost_df['sample_id'] == sample_id]
    else:
        sample_costs = cost_df

    robustness = {}
    for reason_type in ['reasons', 'non_reasons', 'anti_reasons']:
        subset = sample_costs[sample_costs['reason_type'] == reason_type]
        if len(subset) > 0:
            max_cost = subset['cost'].max()
            # Apply the formula: r(C,x) = 1 - max_cost / |features|
            robustness[reason_type] = 1.0 - (max_cost / num_features)
        else:
            robustness[reason_type] = None

    return robustness


def calculate_all_samples_robustness(cost_df: 'pd.DataFrame', num_features: int,
                                     tests_sample: Dict, test_ids: List[str],
                                     verbose: bool = True) -> 'pd.DataFrame':
    """
    Calculate robustness for ALL test samples with prediction information.

    Parameters
    ----------
    cost_df : pd.DataFrame
        DataFrame with costs for all samples
    num_features : int
        Number of features for normalization
    tests_sample : dict
        Dictionary with sample metadata (predicted_label, actual_label, etc.)
    test_ids : list
        List of all test sample IDs
    verbose : bool
        If True, print progress indicators

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: sample_id, robustness, predicted_label,
        actual_label, correct_prediction
    """
    import pandas as pd

    # Filter only anti_reasons for robustness calculation (as per paper)
    anti_reasons_df = cost_df[cost_df['reason_type'] == 'anti_reasons'].copy()

    sample_robustness_results = []

    if verbose:
        print(f"Processing {len(test_ids)} samples...")

    for idx, sample_id in enumerate(test_ids):
        if verbose and (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(test_ids)} samples...")

        rob = calculate_sample_robustness(anti_reasons_df, num_features, sample_id)
        robustness_value = rob.get('anti_reasons')

        # Extract prediction information from tests_sample
        sample_meta = tests_sample.get(sample_id, {})
        predicted_label = sample_meta.get('predicted_label', 'N/A')
        actual_label = sample_meta.get('actual_label', 'N/A')
        is_correct = sample_meta.get('prediction_correct', None)

        sample_robustness_results.append({
            'sample_id': sample_id,
            'robustness': robustness_value,
            'predicted_label': predicted_label,
            'actual_label': actual_label,
            'correct_prediction': is_correct
        })

    if verbose:
        print(f" All {len(test_ids)} samples processed")

    return pd.DataFrame(sample_robustness_results)


def print_robustness_statistics(sample_robustness_df: 'pd.DataFrame') -> None:
    """
    Print comprehensive statistics for sample robustness.

    Parameters
    ----------
    sample_robustness_df : pd.DataFrame
        DataFrame from calculate_all_samples_robustness
    """
    import pandas as pd
    import numpy as np

    print(f"\n\n{'='*80}")
    print(f"  SAMPLE-LEVEL ROBUSTNESS ANALYSIS")
    print(f"{'='*80}\n")

    # Remove samples with no robustness value
    valid_robustness = sample_robustness_df['robustness'].dropna()

    if len(valid_robustness) == 0:
        print("\n WARNING: No valid robustness values found!")
        return

    # Prediction correctness statistics
    correct_count = sample_robustness_df['correct_prediction'].sum()
    total_count = len(sample_robustness_df)
    incorrect_count = total_count - correct_count
    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

    # Basic info
    print(f" Dataset Overview:")
    print(f"   • Total samples:        {total_count}")
    print(f"   • Correct predictions:  {correct_count} ({accuracy:.1f}%)")
    print(f"   • Incorrect predictions: {incorrect_count} ({100-accuracy:.1f}%)")

    # Descriptive statistics - handle NaN for std dev
    mean_val = valid_robustness.mean()
    median_val = valid_robustness.median()
    std_val = valid_robustness.std()
    min_val = valid_robustness.min()
    max_val = valid_robustness.max()
    q25 = valid_robustness.quantile(0.25)
    q75 = valid_robustness.quantile(0.75)
    iqr = q75 - q25

    # Handle NaN in std dev (happens when n=1)
    if np.isnan(std_val) or len(valid_robustness) == 1:
        std_val = 0.0

    print(f"\n Robustness Statistics (All Samples):")
    print(f"   • Mean:      {mean_val:.6f} +/- {std_val:.6f}")
    print(f"   • Median:    {median_val:.6f}")
    print(f"   • Min/Max:   {min_val:.6f} / {max_val:.6f}")
    print(f"   • Range:     {(max_val - min_val):.6f}")
    print(f"   • IQR:       {iqr:.6f} (Q1={q25:.6f}, Q3={q75:.6f})")

    # Distribution analysis
    low_robustness = (valid_robustness < 0.33).sum()
    medium_robustness = ((valid_robustness >= 0.33) & (valid_robustness < 0.67)).sum()
    high_robustness = (valid_robustness >= 0.67).sum()

    # Robustness comparison: Correct vs Incorrect predictions
    correct_samples = sample_robustness_df[sample_robustness_df['correct_prediction'] == True]
    incorrect_samples = sample_robustness_df[sample_robustness_df['correct_prediction'] == False]

    correct_rob = None
    incorrect_rob = None

    print(f"\n Robustness by Prediction Correctness:")
    print(f"{'-'*80}")

    if len(correct_samples) > 0:
        correct_rob = correct_samples['robustness'].dropna()
        if len(correct_rob) > 0:
            c_mean = correct_rob.mean()
            c_std = correct_rob.std()
            c_median = correct_rob.median()
            c_min = correct_rob.min()
            c_max = correct_rob.max()

            # Handle NaN in std dev
            if np.isnan(c_std) or len(correct_rob) == 1:
                c_std = 0.0

            print(f"\n   CORRECT Predictions ({len(correct_samples)} samples):")
            print(f"     Mean:      {c_mean:.6f} +/- {c_std:.6f}")
            print(f"     Median:    {c_median:.6f}")
            print(f"     Range:     [{c_min:.6f}, {c_max:.6f}]")

    if len(incorrect_samples) > 0:
        incorrect_rob = incorrect_samples['robustness'].dropna()
        if len(incorrect_rob) > 0:
            i_mean = incorrect_rob.mean()
            i_std = incorrect_rob.std()
            i_median = incorrect_rob.median()
            i_min = incorrect_rob.min()
            i_max = incorrect_rob.max()

            # Handle NaN in std dev
            if np.isnan(i_std) or len(incorrect_rob) == 1:
                i_std = 0.0

            print(f"\n   INCORRECT Predictions ({len(incorrect_samples)} samples):")
            print(f"     Mean:      {i_mean:.6f} +/- {i_std:.6f}")
            print(f"     Median:    {i_median:.6f}")
            print(f"     Range:     [{i_min:.6f}, {i_max:.6f}]")

            # Calculate difference only if we have correct predictions data
            if correct_rob is not None and len(correct_rob) > 0 and 'c_mean' in locals():
                diff = c_mean - i_mean
                sign = "+" if diff >= 0 else ""
                print(f"\n   Difference (Correct - Incorrect): {sign}{diff:.6f}")

    print(f"\n{'-'*80}")


def create_robustness_visualizations(sample_robustness_df: 'pd.DataFrame',
                                     dataset_name: str = 'Dataset'):
    """
    Create comprehensive visualizations for sample robustness.

    Parameters
    ----------
    sample_robustness_df : pd.DataFrame
        DataFrame from calculate_all_samples_robustness
    dataset_name : str
        Name of the dataset for plot titles

    Returns
    -------
    tuple
        (main_figure, quartile_figure) - Two plotly figures
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import pandas as pd

    valid_rob = sample_robustness_df['robustness'].dropna()

    if len(valid_rob) == 0:
        print("No valid robustness data to visualize")
        return None, None

    # Create subplots: Boxplot, Histogram, and Cumulative Distribution
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Boxplot - Robustness Distribution',
            'Histogram - Robustness Frequency',
            'Cumulative Distribution',
            'Sample Robustness Values'
        ),
        specs=[[{"type": "box"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )

    # 1. Boxplot
    fig.add_trace(go.Box(
        y=valid_rob,
        name='Robustness',
        boxmean='sd',
        marker=dict(color='lightblue'),
        boxpoints='outliers'
    ), row=1, col=1)

    # 2. Histogram with distribution
    fig.add_trace(go.Histogram(
        x=valid_rob,
        nbinsx=30,
        name='Frequency',
        marker=dict(color='steelblue'),
        opacity=0.7
    ), row=1, col=2)

    # Add mean and median lines to histogram
    mean_val = valid_rob.mean()
    median_val = valid_rob.median()

    fig.add_vline(x=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.3f}",
                 annotation_position="top", row=1, col=2)
    fig.add_vline(x=median_val, line_dash="dash", line_color="green",
                 annotation_text=f"Median: {median_val:.3f}",
                 annotation_position="bottom right", row=1, col=2)

    # 3. Cumulative Distribution
    sorted_rob = np.sort(valid_rob)
    cumulative = np.arange(1, len(sorted_rob) + 1) / len(sorted_rob)

    fig.add_trace(go.Scatter(
        x=sorted_rob,
        y=cumulative,
        mode='lines',
        name='CDF',
        line=dict(color='purple', width=2)
    ), row=2, col=1)

    # Add percentile markers
    for percentile in [0.25, 0.5, 0.75]:
        val = np.percentile(valid_rob, percentile * 100)
        fig.add_trace(go.Scatter(
            x=[val],
            y=[percentile],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=[f'{percentile*100:.0f}%: {val:.3f}'],
            textposition="top center",
            showlegend=False
        ), row=2, col=1)

    # 4. Individual Sample Robustness Values (colored by prediction correctness)
    valid_samples_df = sample_robustness_df[sample_robustness_df['robustness'].notna()].copy()

    # Separate correct and incorrect predictions
    correct_mask = valid_samples_df['correct_prediction'] == True
    incorrect_mask = valid_samples_df['correct_prediction'] == False

    # Plot correct predictions
    if correct_mask.sum() > 0:
        correct_df = valid_samples_df[correct_mask]
        fig.add_trace(go.Scatter(
            x=list(range(len(correct_df))),
            y=correct_df['robustness'].values,
            mode='markers',
            name='Correct Predictions',
            marker=dict(size=5, color='green', symbol='circle'),
            text=[f'Sample {sid}<br>Predicted: {pred}<br>Actual: {act}'
                  for sid, pred, act in zip(correct_df['sample_id'],
                                           correct_df['predicted_label'],
                                           correct_df['actual_label'])],
            hovertemplate='%{text}<br>Robustness: %{y:.4f}<extra></extra>',
            legendgroup='correct'
        ), row=2, col=2)

    # Plot incorrect predictions
    if incorrect_mask.sum() > 0:
        incorrect_df = valid_samples_df[incorrect_mask]
        x_offset = correct_mask.sum()
        fig.add_trace(go.Scatter(
            x=list(range(x_offset, x_offset + len(incorrect_df))),
            y=incorrect_df['robustness'].values,
            mode='markers',
            name='Incorrect Predictions',
            marker=dict(size=5, color='red', symbol='x'),
            text=[f'Sample {sid}<br>Predicted: {pred}<br>Actual: {act}'
                  for sid, pred, act in zip(incorrect_df['sample_id'],
                                           incorrect_df['predicted_label'],
                                           incorrect_df['actual_label'])],
            hovertemplate='%{text}<br>Robustness: %{y:.4f}<extra></extra>',
            legendgroup='incorrect'
        ), row=2, col=2)

    # Add mean line
    fig.add_hline(y=mean_val, line_dash="dash", line_color="red",
                 annotation_text=f"Mean: {mean_val:.3f}",
                 row=2, col=2)

    # Update axes labels
    fig.update_xaxes(title_text="Robustness Value", row=1, col=2)
    fig.update_yaxes(title_text="Robustness", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Robustness Value", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1)
    fig.update_xaxes(title_text="Sample Index", row=2, col=2)
    fig.update_yaxes(title_text="Robustness", row=2, col=2)

    fig.update_layout(
        title_text=f"Sample-Level Robustness Analysis - {dataset_name}<br><sub>Total Samples: {len(valid_rob)} | Mean: {mean_val:.4f} ± {valid_rob.std():.4f}</sub>",
        height=900,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.45,
            xanchor="left",
            x=0.52
        )
    )

    # Additional statistics plot: Robustness by quartile
    quartiles = pd.qcut(valid_rob, q=4, labels=False, duplicates='drop')
    quartile_counts = quartiles.value_counts().sort_index()
    q_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
    quartile_counts.index = [q_labels[int(i)] if int(i) < len(q_labels) else f'Q{int(i)+1}' for i in quartile_counts.index]
    colors = ['red', 'orange', 'lightgreen', 'green']

    fig2 = go.Figure(data=[
        go.Bar(
            x=quartile_counts.index.astype(str),
            y=quartile_counts.values,
            text=quartile_counts.values,
            textposition='auto',
            marker=dict(color=colors[:len(quartile_counts)])
        )
    ])

    fig2.update_layout(
        title=f'Sample Distribution by Robustness Quartile - {dataset_name}',
        xaxis_title='Quartile',
        yaxis_title='Number of Samples',
        template='plotly_white',
        height=400
    )

    return fig, fig2


def visualize_all_time_series(tests_sample: Dict, test_ids: List[str],
                              feature_names: List[str], max_samples: int = 50):
    """
    Visualize all time series samples.

    Parameters
    ----------
    tests_sample : dict
        Dictionary with sample data
    test_ids : list
        List of test sample IDs
    feature_names : list
        List of feature names
    max_samples : int
        Maximum number of samples to plot

    Returns
    -------
    plotly.Figure
        Figure with all time series
    """
    import plotly.graph_objects as go
    import numpy as np

    samples_to_plot = test_ids[:min(max_samples, len(test_ids))]

    fig = go.Figure()

    for idx, sample_id in enumerate(samples_to_plot):
        sample_dict = tests_sample[sample_id]["features"]
        series = np.array([sample_dict[f] for f in feature_names])
        x_axis = np.arange(len(series))

        fig.add_trace(go.Scatter(
            x=x_axis,
            y=series,
            mode='lines',
            name=f'Sample {sample_id}',
            line=dict(width=1.5),
            opacity=0.7,
            showlegend=(len(samples_to_plot) <= 20)
        ))

    fig.update_layout(
        title=f'Time Series - All Test Samples ({len(samples_to_plot)} samples)',
        xaxis_title='Feature Index',
        yaxis_title='Value',
        template='plotly_white',
        height=600,
        showlegend=True if len(samples_to_plot) <= 20 else False,
        hovermode='closest'
    )

    return fig


def visualize_sample_with_icf(sample_id: str, tests_sample: Dict,
                              feature_names: List[str], reason_type: str = 'reasons'):
    """
    Visualize a single sample with its ICF constraints.

    Parameters
    ----------
    sample_id : str
        Sample ID to visualize
    tests_sample : dict
        Dictionary with sample data
    feature_names : list
        List of feature names
    reason_type : str
        Type of reason: 'reasons', 'non_reasons', or 'anti_reasons'

    Returns
    -------
    plotly.Figure or None
        Figure with sample and ICF, or None if no data
    """
    import plotly.graph_objects as go
    import numpy as np

    if reason_type not in tests_sample[sample_id] or len(tests_sample[sample_id][reason_type]) == 0:
        print(f"No {reason_type} found for this sample")
        return None

    # Get sample data
    sample_dict = tests_sample[sample_id]["features"]
    series = np.array([sample_dict[f] for f in feature_names])
    x_axis = np.arange(len(series))
    display_feature_names = [
        _format_feature_label(name, idx) for idx, name in enumerate(feature_names)
    ]

    # Get prediction info
    sample_meta = tests_sample.get(sample_id, {})
    predicted_label = sample_meta.get('predicted_label', 'N/A')
    actual_label = sample_meta.get('actual_label', 'N/A')
    is_correct = sample_meta.get('prediction_correct', None)

    prediction_status = ""
    prediction_symbol = ""
    if is_correct is not None:
        if is_correct:
            prediction_status = f"Predicted={predicted_label}, Actual={actual_label}"
            prediction_symbol = " CORRECT"
        else:
            prediction_status = f"Predicted={predicted_label}, Actual={actual_label}"
            prediction_symbol = " INCORRECT"

    # Get first ICF
    first_bitmap = list(tests_sample[sample_id][reason_type].keys())[0]
    icf = tests_sample[sample_id][reason_type][first_bitmap]["icf"]
    cost = tests_sample[sample_id][reason_type][first_bitmap]["cost"]

    # Determine color based on reason type
    colors = {
        'reasons': ('rgba(0, 255, 0, 0.15)', 'green', 'darkgreen'),
        'non_reasons': ('rgba(255, 165, 0, 0.15)', 'orange', 'darkorange'),
        'anti_reasons': ('rgba(255, 0, 0, 0.15)', 'red', 'darkred')
    }
    fill_color, line_color, dark_color = colors.get(reason_type, ('rgba(0, 0, 255, 0.15)', 'blue', 'darkblue'))

    # Find constrained features and identify contiguous intervals
    constrained_indices = []
    constraint_values = {}

    for idx, f in enumerate(feature_names):
        if f in icf:
            lower_bound, upper_bound = icf[f]
            if not (np.isinf(lower_bound) and np.isinf(upper_bound)):
                constrained_indices.append(idx)
                constraint_values[idx] = {
                    'lower': lower_bound if not np.isinf(lower_bound) else series[idx] - 1.5,
                    'upper': upper_bound if not np.isinf(upper_bound) else series[idx] + 1.5,
                    'lower_inf': np.isinf(lower_bound),
                    'upper_inf': np.isinf(upper_bound),
                    'feature': display_feature_names[idx]
                }

    # Identify contiguous temporal intervals
    temporal_intervals = []
    if constrained_indices:
        constrained_indices.sort()
        start_idx = constrained_indices[0]
        end_idx = constrained_indices[0]

        for i in range(1, len(constrained_indices)):
            if constrained_indices[i] == end_idx + 1:
                end_idx = constrained_indices[i]
            else:
                temporal_intervals.append((start_idx, end_idx))
                start_idx = constrained_indices[i]
                end_idx = constrained_indices[i]
        temporal_intervals.append((start_idx, end_idx))

    # Create figure
    fig = go.Figure()

    # Add colored rectangles for temporal intervals (background) - NO annotations
    for interval_idx, (start, end) in enumerate(temporal_intervals):
        fig.add_vrect(
            x0=start - 0.5, x1=end + 0.5,
            fillcolor=fill_color,
            line_width=0,  # No border line
            line_color=line_color
        )

    # Add time series line
    fig.add_trace(go.Scatter(
        x=x_axis, y=series,
        mode='lines+markers',
        name='Test Sample',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color='blue')
    ))

    # Add constraint intervals as vertical bars
    for idx in constrained_indices:
        constraint_info = constraint_values[idx]
        lower_bound = constraint_info['lower']
        upper_bound = constraint_info['upper']
        feature_label = constraint_info['feature']

        fig.add_trace(go.Scatter(
            x=[idx, idx],
            y=[lower_bound, upper_bound],
            mode='lines+markers',
            line=dict(color=line_color, width=3),
            marker=dict(size=6, color=dark_color, symbol='line-ew-open'),
            showlegend=False,
            hovertemplate=f'<b>{feature_label}</b><br>Y-Constraint: [{lower_bound:.3f}, {upper_bound:.3f}]<br>Value: {series[idx]:.3f}<extra></extra>'
        ))

    reason_label = reason_type.upper().replace('_', '-')

    # Build title with prediction status
    title_text = f'Time Series with Maximal {reason_label} - Sample {sample_id}'
    subtitle_parts = [
        f'Cost: {cost:.4f}',
        f'Constrained: {len(constrained_indices)}/{len(feature_names)}',
        f'Temporal Intervals: {len(temporal_intervals)}'
    ]
    if prediction_symbol:
        subtitle_parts.insert(0, f'{prediction_symbol} - {prediction_status}')

    # Calculate y-range for positioning interval markers below the plot
    y_min = series.min()
    y_max = series.max()
    y_range = y_max - y_min
    marker_y = y_min - 0.15 * y_range  # Position below the minimum value

    # Add interval markers (|---|) below the x-axis
    interval_annotations = []
    for interval_idx, (start, end) in enumerate(temporal_intervals):
        start_feature = display_feature_names[start]
        end_feature = display_feature_names[end]

        # Add vertical bars at start and end
        fig.add_trace(go.Scatter(
            x=[start, start],
            y=[marker_y - 0.02 * y_range, marker_y + 0.02 * y_range],
            mode='lines',
            line=dict(color=dark_color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=[end, end],
            y=[marker_y - 0.02 * y_range, marker_y + 0.02 * y_range],
            mode='lines',
            line=dict(color=dark_color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add horizontal line connecting them
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[marker_y, marker_y],
            mode='lines',
            line=dict(color=dark_color, width=2),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add text annotation for the interval
        interval_text = f"{start_feature}" if start == end else f"{start_feature}→{end_feature}"
        interval_annotations.append(
            dict(
                x=(start + end) / 2,
                y=marker_y - 0.06 * y_range,
                text=interval_text,
                showarrow=False,
                font=dict(size=9, color=dark_color),
                xanchor='center',
                yanchor='top'
            )
        )

    fig.update_layout(
        title=f'{title_text}<br><sub>{" | ".join(subtitle_parts)}</sub>',
        xaxis_title='Feature Index (Time Points)',
        yaxis_title='Value',
        template='plotly_white',
        height=600,
        showlegend=True,
        hovermode='closest',
        annotations=interval_annotations
    )

    # Print summary
    print(f"\n{reason_label} Analysis for Sample {sample_id}:")
    if prediction_symbol:
        print(f"  {prediction_symbol}: {prediction_status}")
    print(f"  Cost: {cost:.6f}")
    print(f"  Constrained features: {len(constrained_indices)}/{len(feature_names)}")
    print(f"  Temporal intervals: {len(temporal_intervals)}")

    return fig


def visualize_anti_reason_corridor(sample_id: str, tests_sample: Dict, feature_names: List[str]):
    """
    Visualize only the anti-reason with a two-color corridor system:
    - Green when sample is within the constraint corridor
    - Red when sample is outside the constraint corridor

    Parameters
    ----------
    sample_id : str
        Sample ID to visualize
    tests_sample : dict
        Dictionary with sample data
    feature_names : list
        List of feature names

    Returns
    -------
    plotly.Figure or None
        Figure with anti-reason corridor colored by constraint satisfaction, or None if missing data
    """
    import plotly.graph_objects as go
    import numpy as np

    # Check if anti_reasons exist
    if ("anti_reasons" not in tests_sample[sample_id] or len(tests_sample[sample_id]["anti_reasons"]) == 0):
        print("No anti_reasons found for this sample")
        return None

    # Get sample data
    sample_dict = tests_sample[sample_id]["features"]
    series = np.array([sample_dict[f] for f in feature_names])
    x_axis = np.arange(len(series))
    display_feature_names = [
        _format_feature_label(name, idx) for idx, name in enumerate(feature_names)
    ]

    # Get prediction info
    sample_meta = tests_sample.get(sample_id, {})
    predicted_label = sample_meta.get('predicted_label', 'N/A')
    actual_label = sample_meta.get('actual_label', 'N/A')
    is_correct = sample_meta.get('prediction_correct', None)

    prediction_status = ""
    if is_correct is not None:
        if is_correct:
            prediction_status = f" CORRECT: Predicted={predicted_label}, Actual={actual_label}"
        else:
            prediction_status = f" INCORRECT: Predicted={predicted_label}, Actual={actual_label}"

    # Get anti-reason data
    first_ar_bitmap = list(tests_sample[sample_id]["anti_reasons"].keys())[0]
    ar_icf = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["icf"]
    ar_cost = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["cost"]

    # Create constraint bounds and check satisfaction for each point
    upper_bounds = []
    lower_bounds = []
    within_constraints = []
    constrained_features = []
    
    for idx, f in enumerate(feature_names):
        if f in ar_icf:
            lower_bound, upper_bound = ar_icf[f]
            if not (np.isinf(lower_bound) and np.isinf(upper_bound)):
                # Use actual constraint bounds
                lower_val = lower_bound if not np.isinf(lower_bound) else series[idx] - 1.0
                upper_val = upper_bound if not np.isinf(upper_bound) else series[idx] + 1.0
                constrained_features.append(True)
                
                # Check if sample point is within constraints
                is_within = lower_val <= series[idx] <= upper_val
                within_constraints.append(is_within)
            else:
                # Unconstrained - always within
                lower_val = series[idx] - 0.5
                upper_val = series[idx] + 0.5
                constrained_features.append(False)
                within_constraints.append(True)
        else:
            # Unconstrained - always within
            lower_val = series[idx] - 0.5
            upper_val = series[idx] + 0.5
            constrained_features.append(False)
            within_constraints.append(True)
        
        lower_bounds.append(lower_val)
        upper_bounds.append(upper_val)
    
    upper_bounds = np.array(upper_bounds)
    lower_bounds = np.array(lower_bounds)
    within_constraints = np.array(within_constraints)
    constrained_features = np.array(constrained_features)

    # Calculate statistics
    total_constrained = np.sum(constrained_features)
    within_count = np.sum(within_constraints & constrained_features)
    violation_count = np.sum(~within_constraints & constrained_features)
    satisfaction_rate = (within_count / total_constrained * 100) if total_constrained > 0 else 100

    # Create figure
    fig = go.Figure()

    # Create segments for different colors based on constraint satisfaction
    # We'll create small segments between each point to color them appropriately
    
    for i in range(len(x_axis) - 1):
        # Create a small segment between point i and i+1
        x_segment = [x_axis[i], x_axis[i+1]]
        upper_segment = [upper_bounds[i], upper_bounds[i+1]]
        lower_segment = [lower_bounds[i], lower_bounds[i+1]]
        
        # Determine color based on constraint satisfaction of the current point
        # Use the more restrictive condition (if either point violates, color it red)
        is_within_segment = within_constraints[i] and within_constraints[i+1]
        is_constrained_segment = constrained_features[i] or constrained_features[i+1]
        
        if is_constrained_segment:
            if is_within_segment:
                # Green for within constraints
                fill_color = 'rgba(0, 255, 0, 0.3)'
                line_color = 'green'
            else:
                # Red for constraint violations
                fill_color = 'rgba(255, 0, 0, 0.3)'
                line_color = 'red'
        else:
            # Light gray for unconstrained regions
            fill_color = 'rgba(66, 66, 66, 0.1)'
            line_color = 'gray'
        
        # Add filled area for this segment
        fig.add_trace(go.Scatter(
            x=[x_segment[0], x_segment[1], x_segment[1], x_segment[0]],
            y=[lower_segment[0], lower_segment[1], upper_segment[1], upper_segment[0]],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color='rgba(255,255,255,0)', width=0),
            hoverinfo="skip",
            showlegend=False
        ))

    # Add legend entries for corridor colors
    # Add dummy traces for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=16, color='rgba(0, 255, 0, 0.6)'),
        name='Sample in ICF',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers', 
        marker=dict(size=16, color='rgba(255, 0, 0, 0.6)'),
        name='Sample out of ICF',
        showlegend=True
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=16, color='rgba(66, 66, 66, 0.3)'),
        name='Do not care (Unconstrained)',
        showlegend=True
    ))

    # Add the sample time series line (dark gray)
    fig.add_trace(go.Scatter(
        x=x_axis, 
        y=series,
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(size=5, color='blue'),
        name='Test Sample',
        hovertemplate='Feature %{x}<br>Value: %{y:.3f}<extra></extra>'
    ))

    # Parse sample ID to extract dataset, class, and sample index
    sample_parts = sample_id.split('_')
    if len(sample_parts) >= 3:
        dataset_name = sample_parts[0]
        class_label = sample_parts[1] 
        sample_index = sample_parts[2]
        title_main = f'Anti-Reason Analysis - Dataset {dataset_name}, Class {class_label}, Sample Index {sample_index}, Predicted={predicted_label}'
    else:
        # Fallback if sample ID doesn't follow expected format
        title_main = f'Anti-Reason Analysis - Sample {sample_id}'
        
    
    
    subtitle_parts = [
        f'Cost: {ar_cost:.2f}',
        f'ICF Features: {total_constrained}/{len(feature_names)}',
        f'Sample in ICF Rate: {satisfaction_rate:.1f}% ({within_count}/{total_constrained})',
        f'Sample Out of ICF: {violation_count/total_constrained*100 if total_constrained > 0 else 0.0:.1f}% ({violation_count}/{total_constrained})'
    ]
    # if prediction_status:
    #     subtitle_parts.append(f'<br><sub>{prediction_status}</sub>')
    title_main += f'<br><sub>{" | ".join(subtitle_parts)}</sub>'

    # Update layout
    fig.update_layout(
        title=title_main,
        xaxis_title="Feature Index (Time Points)",
        yaxis_title="Value",
        template='plotly_white',
        height=700,
        showlegend=True,
        hovermode='closest',
        font=dict(size=20, family='Arial, sans-serif'),
        title_font_size=28,
        legend=dict(
            yanchor="bottom",
            y=0.02,
            xanchor="left",
            x=0.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(0,0,0,0.3)",
            borderwidth=2,
            font=dict(size=18)
        ),
        xaxis=dict(
            title_font_size=32,
            tickfont_size=28
        ),
        yaxis=dict(
            title_font_size=32,
            tickfont_size=28
        )
    )

    # Print detailed summary
    print(f"\nAnti-Reason Constraint Analysis for Sample {sample_id}:")
    if prediction_status:
        print(f"  {prediction_status}")
    print(f"\n  CONSTRAINT SATISFACTION:")
    print(f"    Cost: {ar_cost:.6f}")
    print(f"    Total features: {len(feature_names)}")
    print(f"    Constrained features: {total_constrained}")
    print(f"    Unconstrained features: {len(feature_names) - total_constrained}")
    print(f"     Within constraints: {within_count}/{total_constrained} ({satisfaction_rate:.1f}%)")
    print(f"     Constraint violations: {violation_count}/{total_constrained}")
    
    if violation_count > 0:
        violation_indices = np.where(~within_constraints & constrained_features)[0]
        violation_features = [display_feature_names[i] for i in violation_indices]
        print(f"    Violated features: {', '.join(violation_features[:5])}")
        if len(violation_features) > 5:
            print(f"      ... and {len(violation_features) - 5} more")

    return fig


def visualize_sample_comparison_smooth(sample_id: str, tests_sample: Dict, feature_names: List[str]):
    """
    Compare reason and anti-reason for a single sample with smooth corridor visualization.
    Instead of showing discrete interval points, this shows smooth constraint corridors.

    Parameters
    ----------
    sample_id : str
        Sample ID to visualize
    tests_sample : dict
        Dictionary with sample data
    feature_names : list
        List of feature names

    Returns
    -------
    plotly.Figure or None
        Combined figure with smooth reason and anti-reason corridors, or None if missing data
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np

    # Check if both reasons and anti_reasons exist
    if ("reasons" not in tests_sample[sample_id] or len(tests_sample[sample_id]["reasons"]) == 0 or
        "anti_reasons" not in tests_sample[sample_id] or len(tests_sample[sample_id]["anti_reasons"]) == 0):
        print("Need both reasons and anti_reasons for comparison")
        return None

    # Get sample data
    sample_dict = tests_sample[sample_id]["features"]
    series = np.array([sample_dict[f] for f in feature_names])
    x_axis = np.arange(len(series))
    display_feature_names = [
        _format_feature_label(name, idx) for idx, name in enumerate(feature_names)
    ]

    # Get prediction info
    sample_meta = tests_sample.get(sample_id, {})
    predicted_label = sample_meta.get('predicted_label', 'N/A')
    actual_label = sample_meta.get('actual_label', 'N/A')
    is_correct = sample_meta.get('prediction_correct', None)

    prediction_status = ""
    if is_correct is not None:
        if is_correct:
            prediction_status = f" CORRECT: Predicted={predicted_label}, Actual={actual_label}"
        else:
            prediction_status = f" INCORRECT: Predicted={predicted_label}, Actual={actual_label}"

    # Get reason data
    first_reason_bitmap = list(tests_sample[sample_id]["reasons"].keys())[0]
    reason_icf = tests_sample[sample_id]["reasons"][first_reason_bitmap]["icf"]
    reason_cost = tests_sample[sample_id]["reasons"][first_reason_bitmap]["cost"]

    # Get anti-reason data
    first_ar_bitmap = list(tests_sample[sample_id]["anti_reasons"].keys())[0]
    ar_icf = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["icf"]
    ar_cost = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["cost"]

    def create_smooth_corridor(icf_data, feature_names_list, series_data, x_values):
        """Create smooth upper and lower bounds for the constraint corridor"""
        upper_bounds = []
        lower_bounds = []
        constrained_count = 0
        
        for idx, f in enumerate(feature_names_list):
            if f in icf_data:
                lower_bound, upper_bound = icf_data[f]
                if not (np.isinf(lower_bound) and np.isinf(upper_bound)):
                    # Use actual constraint bounds
                    lower_val = lower_bound if not np.isinf(lower_bound) else series_data[idx] - 1.0
                    upper_val = upper_bound if not np.isinf(upper_bound) else series_data[idx] + 1.0
                    constrained_count += 1
                else:
                    # Unconstrained - use series value with small margin
                    lower_val = series_data[idx] - 0.1
                    upper_val = series_data[idx] + 0.1
            else:
                # Unconstrained - use series value with small margin
                lower_val = series_data[idx] - 0.1
                upper_val = series_data[idx] + 0.1
            
            lower_bounds.append(lower_val)
            upper_bounds.append(upper_val)
        
        return np.array(lower_bounds), np.array(upper_bounds), constrained_count

    # Create smooth corridors for both reason and anti-reason
    reason_lower, reason_upper, reason_constrained = create_smooth_corridor(reason_icf, feature_names, series, x_axis)
    ar_lower, ar_upper, ar_constrained = create_smooth_corridor(ar_icf, feature_names, series, x_axis)

    # Create subplots
    title_main = f'Smooth Corridor Comparison: Reason vs Anti-Reason - Sample {sample_id}'
    if prediction_status:
        title_main += f'<br><sub>{prediction_status}</sub>'

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Maximal REASON (cost={reason_cost:.4f}, {reason_constrained}/{len(feature_names)} constrained)',
            f'Maximal ANTI-REASON (cost={ar_cost:.4f}, {ar_constrained}/{len(feature_names)} constrained)'
        ),
        vertical_spacing=0.15
    )

    # Top plot: Reason corridor
    # Add filled area between upper and lower bounds
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_axis, x_axis[::-1]]),
        y=np.concatenate([reason_upper, reason_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Reason Corridor'
    ), row=1, col=1)

    # Add upper and lower bound lines for reason
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=reason_upper,
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='Upper Bound',
        hovertemplate='Feature %{x}<br>Upper Bound: %{y:.3f}<extra></extra>',
        showlegend=True
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=reason_lower,
        mode='lines',
        line=dict(color='green', width=2, dash='dot'),
        name='Lower Bound',
        hovertemplate='Feature %{x}<br>Lower Bound: %{y:.3f}<extra></extra>',
        showlegend=True
    ), row=1, col=1)

    # Add time series line for reason
    fig.add_trace(go.Scatter(
        x=x_axis, 
        y=series,
        mode='lines+markers',
        name='Test Sample',
        line=dict(color='blue', width=3),
        marker=dict(size=5, color='blue'),
        hovertemplate='Feature %{x}<br>Value: %{y:.3f}<extra></extra>',
        showlegend=True
    ), row=1, col=1)

    # Bottom plot: Anti-Reason corridor
    # Add filled area between upper and lower bounds
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_axis, x_axis[::-1]]),
        y=np.concatenate([ar_upper, ar_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name='Anti-Reason Corridor'
    ), row=2, col=1)

    # Add upper and lower bound lines for anti-reason
    fig.add_trace(go.Scatter(
        x=x_axis,
        y=ar_upper,
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Upper Bound',
        hovertemplate='Feature %{x}<br>Upper Bound: %{y:.3f}<extra></extra>',
        showlegend=False
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=x_axis,
        y=ar_lower,
        mode='lines',
        line=dict(color='red', width=2, dash='dot'),
        name='Lower Bound',
        hovertemplate='Feature %{x}<br>Lower Bound: %{y:.3f}<extra></extra>',
        showlegend=False
    ), row=2, col=1)

    # Add time series line for anti-reason
    fig.add_trace(go.Scatter(
        x=x_axis, 
        y=series,
        mode='lines+markers',
        name='Test Sample',
        line=dict(color='blue', width=3),
        marker=dict(size=5, color='blue'),
        hovertemplate='Feature %{x}<br>Value: %{y:.3f}<extra></extra>',
        showlegend=False
    ), row=2, col=1)

    # Update axes
    fig.update_xaxes(title_text="Feature Index (Time Points)", row=1, col=1)
    fig.update_xaxes(title_text="Feature Index (Time Points)", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    # Update layout
    fig.update_layout(
        title=title_main,
        height=950,
        template='plotly_white',
        showlegend=True,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.48,
            xanchor="left",
            x=1.02
        )
    )

    # Print summary
    print(f"\nSmooth Corridor Comparison for Sample {sample_id}:")
    if prediction_status:
        print(f"  {prediction_status}")
    print(f"\n  REASON CORRIDOR:")
    print(f"    Cost: {reason_cost:.6f}")
    print(f"    Constrained features: {reason_constrained}/{len(feature_names)}")
    print(f"    Corridor width: avg={np.mean(reason_upper - reason_lower):.3f}")
    print(f"\n  ANTI-REASON CORRIDOR:")
    print(f"    Cost: {ar_cost:.6f}")
    print(f"    Constrained features: {ar_constrained}/{len(feature_names)}")
    print(f"    Corridor width: avg={np.mean(ar_upper - ar_lower):.3f}")

    return fig


def visualize_sample_comparison(sample_id: str, tests_sample: Dict, feature_names: List[str]):
    """
    Compare reason and anti-reason for a single sample.

    Parameters
    ----------
    sample_id : str
        Sample ID to visualize
    tests_sample : dict
        Dictionary with sample data
    feature_names : list
        List of feature names

    Returns
    -------
    plotly.Figure or None
        Combined figure with reason and anti-reason, or None if missing data
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import numpy as np

    # Check if both reasons and anti_reasons exist
    if ("reasons" not in tests_sample[sample_id] or len(tests_sample[sample_id]["reasons"]) == 0 or
        "anti_reasons" not in tests_sample[sample_id] or len(tests_sample[sample_id]["anti_reasons"]) == 0):
        print("Need both reasons and anti_reasons for comparison")
        return None

    # Get sample data
    sample_dict = tests_sample[sample_id]["features"]
    series = np.array([sample_dict[f] for f in feature_names])
    x_axis = np.arange(len(series))
    display_feature_names = [
        _format_feature_label(name, idx) for idx, name in enumerate(feature_names)
    ]

    # Get prediction info
    sample_meta = tests_sample.get(sample_id, {})
    predicted_label = sample_meta.get('predicted_label', 'N/A')
    actual_label = sample_meta.get('actual_label', 'N/A')
    is_correct = sample_meta.get('prediction_correct', None)

    prediction_status = ""
    if is_correct is not None:
        if is_correct:
            prediction_status = f" CORRECT: Predicted={predicted_label}, Actual={actual_label}"
        else:
            prediction_status = f" INCORRECT: Predicted={predicted_label}, Actual={actual_label}"

    # Get reason data
    first_reason_bitmap = list(tests_sample[sample_id]["reasons"].keys())[0]
    reason_icf = tests_sample[sample_id]["reasons"][first_reason_bitmap]["icf"]
    reason_cost = tests_sample[sample_id]["reasons"][first_reason_bitmap]["cost"]

    # Get anti-reason data
    first_ar_bitmap = list(tests_sample[sample_id]["anti_reasons"].keys())[0]
    ar_icf = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["icf"]
    ar_cost = tests_sample[sample_id]["anti_reasons"][first_ar_bitmap]["cost"]

    # Helper function to find contiguous intervals
    def find_contiguous_intervals(icf_data, feature_names_list, series_data):
        constrained_indices = []
        constraint_values = {}

        for idx, f in enumerate(feature_names_list):
            if f in icf_data:
                lower_bound, upper_bound = icf_data[f]
                if not (np.isinf(lower_bound) and np.isinf(upper_bound)):
                    constrained_indices.append(idx)
                    constraint_values[idx] = {
                        'lower': lower_bound if not np.isinf(lower_bound) else series_data[idx] - 1.5,
                        'upper': upper_bound if not np.isinf(upper_bound) else series_data[idx] + 1.5,
                        'feature': display_feature_names[idx]
                    }

        temporal_intervals = []
        if constrained_indices:
            constrained_indices.sort()
            start_idx = constrained_indices[0]
            end_idx = constrained_indices[0]

            for i in range(1, len(constrained_indices)):
                if constrained_indices[i] == end_idx + 1:
                    end_idx = constrained_indices[i]
                else:
                    temporal_intervals.append((start_idx, end_idx))
                    start_idx = constrained_indices[i]
                    end_idx = constrained_indices[i]
            temporal_intervals.append((start_idx, end_idx))

        return constrained_indices, constraint_values, temporal_intervals

    # Process reason intervals
    reason_indices, reason_constraints, reason_intervals = find_contiguous_intervals(reason_icf, feature_names, series)

    # Process anti-reason intervals
    ar_indices, ar_constraints, ar_intervals = find_contiguous_intervals(ar_icf, feature_names, series)

    # Create subplots
    title_main = f'Comparison: Reason vs Anti-Reason - Sample {sample_id}'
    if prediction_status:
        title_main += f'<br><sub>{prediction_status}</sub>'

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Maximal REASON (cost={reason_cost:.4f}, {len(reason_intervals)} intervals)',
            f'Maximal ANTI-REASON (cost={ar_cost:.4f}, {len(ar_intervals)} intervals)'
        ),
        vertical_spacing=0.15
    )

    # Top plot: Reason with temporal intervals
    # Add colored rectangles for temporal intervals (NO annotations)
    for start, end in reason_intervals:
        fig.add_vrect(
            x0=start - 0.5, x1=end + 0.5,
            fillcolor='rgba(0, 255, 0, 0.15)',
            line_width=0,
            line_color='green',
            row=1, col=1
        )

    fig.add_trace(go.Scatter(
        x=x_axis, y=series,
        mode='lines+markers',
        name='Test Sample',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color='blue'),
        showlegend=True
    ), row=1, col=1)

    for idx in reason_indices:
        constraint_info = reason_constraints[idx]
        lower_bound = constraint_info['lower']
        upper_bound = constraint_info['upper']
        feature_label = constraint_info['feature']

        fig.add_trace(go.Scatter(
            x=[idx, idx],
            y=[lower_bound, upper_bound],
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=6, color='darkgreen', symbol='line-ew-open'),
            showlegend=False,
            hovertemplate=f'<b>{feature_label}</b><br>Y-Constraint: [{lower_bound:.3f}, {upper_bound:.3f}]<extra></extra>'
        ), row=1, col=1)

    # Bottom plot: Anti-Reason with temporal intervals
    # Add colored rectangles for temporal intervals (NO annotations)
    for start, end in ar_intervals:
        fig.add_vrect(
            x0=start - 0.5, x1=end + 0.5,
            fillcolor='rgba(255, 0, 0, 0.15)',
            line_width=0,
            line_color='red',
            row=2, col=1
        )

    fig.add_trace(go.Scatter(
        x=x_axis, y=series,
        mode='lines+markers',
        name='Test Sample',
        line=dict(color='blue', width=2),
        marker=dict(size=4, color='blue'),
        showlegend=False
    ), row=2, col=1)

    for idx in ar_indices:
        constraint_info = ar_constraints[idx]
        lower_bound = constraint_info['lower']
        upper_bound = constraint_info['upper']
        feature_label = constraint_info['feature']

        fig.add_trace(go.Scatter(
            x=[idx, idx],
            y=[lower_bound, upper_bound],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='darkred', symbol='line-ew-open'),
            showlegend=False,
            hovertemplate=f'<b>{feature_label}</b><br>Y-Constraint: [{lower_bound:.3f}, {upper_bound:.3f}]<extra></extra>'
        ), row=2, col=1)

    # Calculate y-range for positioning interval markers
    y_min = series.min()
    y_max = series.max()
    y_range = y_max - y_min

    # Add interval markers for REASON (top plot)
    marker_y_top = y_min - 0.15 * y_range
    reason_annotations = []
    for start, end in reason_intervals:
        start_feature = display_feature_names[start]
        end_feature = display_feature_names[end]

        # Vertical bars
        fig.add_trace(go.Scatter(
            x=[start, start],
            y=[marker_y_top - 0.02 * y_range, marker_y_top + 0.02 * y_range],
            mode='lines',
            line=dict(color='darkgreen', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[end, end],
            y=[marker_y_top - 0.02 * y_range, marker_y_top + 0.02 * y_range],
            mode='lines',
            line=dict(color='darkgreen', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

        # Horizontal line
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[marker_y_top, marker_y_top],
            mode='lines',
            line=dict(color='darkgreen', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=1)

        # Text annotation
        interval_text = f"{start_feature}" if start == end else f"{start_feature}→{end_feature}"
        reason_annotations.append(
            dict(
                x=(start + end) / 2,
                y=marker_y_top - 0.06 * y_range,
                text=interval_text,
                showarrow=False,
                font=dict(size=9, color='darkgreen'),
                xanchor='center',
                yanchor='top',
                xref='x1',
                yref='y1'
            )
        )

    # Add interval markers for ANTI-REASON (bottom plot)
    marker_y_bottom = y_min - 0.15 * y_range
    ar_annotations = []
    for start, end in ar_intervals:
        start_feature = display_feature_names[start]
        end_feature = display_feature_names[end]

        # Vertical bars
        fig.add_trace(go.Scatter(
            x=[start, start],
            y=[marker_y_bottom - 0.02 * y_range, marker_y_bottom + 0.02 * y_range],
            mode='lines',
            line=dict(color='darkred', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=[end, end],
            y=[marker_y_bottom - 0.02 * y_range, marker_y_bottom + 0.02 * y_range],
            mode='lines',
            line=dict(color='darkred', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=1)

        # Horizontal line
        fig.add_trace(go.Scatter(
            x=[start, end],
            y=[marker_y_bottom, marker_y_bottom],
            mode='lines',
            line=dict(color='darkred', width=2),
            showlegend=False,
            hoverinfo='skip'
        ), row=2, col=1)

        # Text annotation
        interval_text = f"{start_feature}" if start == end else f"{start_feature}→{end_feature}"
        ar_annotations.append(
            dict(
                x=(start + end) / 2,
                y=marker_y_bottom - 0.06 * y_range,
                text=interval_text,
                showarrow=False,
                font=dict(size=9, color='darkred'),
                xanchor='center',
                yanchor='top',
                xref='x2',
                yref='y2'
            )
        )

    fig.update_xaxes(title_text="Feature Index (Time Points)", row=1, col=1)
    fig.update_xaxes(title_text="Feature Index (Time Points)", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    fig.update_layout(
        title=title_main,
        height=950,
        template='plotly_white',
        showlegend=True,
        hovermode='closest',
        annotations=reason_annotations + ar_annotations
    )

    print(f"\nComparison for Sample {sample_id}:")
    if prediction_status:
        print(f"  {prediction_status}")
    print(f"\n  REASON:")
    print(f"    Constrained features: {len(reason_indices)}")
    print(f"    Temporal intervals: {len(reason_intervals)}")
    print(f"  ANTI-REASON:")
    print(f"    Constrained features: {len(ar_indices)}")
    print(f"    Temporal intervals: {len(ar_intervals)}")

    return fig

import json
import math
import os
import statistics
import zipfile
from collections import defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
try:
    from IPython.core.display_functions import display
except ImportError:
    display = print
from matplotlib import pyplot as plt, dates as mdates

from etl.constants import DB_NAMES

try:
	import ipywidgets as widgets
except ImportError:
	widgets = None

from etl.zip_inspector import try_decode_value, decode_key, CANDIDATE_ANTI_REASONS_NAME, CANDIDATE_REASONS_NAME, \
	DB_DISPLAY_NAMES, BASE_COLUMNS, SCATTER_PLOTS, ADDITIONAL_SCATTER_PREFIX_PAIRS, BAR_PLOTS, STACKED_BAR_CONFIG, \
	HISTOGRAMS


def summarise_db10_workers(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, *, show_summary=True):
    if not selected_zip_name:
        raise ValueError("No archive selected.")

    files_map = (selected_manifest or {}).get('files', {}) or {}
    db10_file = files_map.get('10')
    if not db10_file:
        raise ValueError("No DB 10 available for the current selection.")

    backups = selected_backups if isinstance(selected_backups, dict) else {}
    db10_data = backups.get(db10_file)
    if not isinstance(db10_data, dict):
        zip_path_str = selected_archive_data.get('zip_path') if selected_archive_data else None
        if zip_path_str:
            zip_path = Path(zip_path_str)
            if zip_path.exists():
                try:
                    with zipfile.ZipFile(zip_path) as archive:
                        payload = archive.read(f"{selected_manifest_prefix}{db10_file}")
                    db10_data = json.loads(payload.decode('utf-8', errors='replace'))
                except Exception:
                    db10_data = None
    if not isinstance(db10_data, dict):
        raise ValueError("Unable to load DB 10 data.")

    entries = db10_data.get('entries') or []
    if not entries:
        print("No entries available in DB 10.")
        return {
            'zip_name': selected_zip_name,
            'db10_file': db10_file,
            'entries': entries,
            'db10_data': db10_data,
            'worker_stats': {},
            'worker_summaries': [],
        }

    worker_stats = {}
    for entry in entries:
        preview, details = try_decode_value(entry)
        raw_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None
        if isinstance(raw_bytes, (bytes, bytearray)):
            text = raw_bytes.decode('utf-8', errors='replace')
        elif isinstance(preview, (bytes, bytearray)):
            text = preview.decode('utf-8', errors='replace')
        else:
            text = str(preview)
        try:
            payload = json.loads(text)
        except Exception:
            continue
        worker_id = payload.get('worker_id')
        if not worker_id:
            try:
                key_text = decode_key(entry).decode('utf-8', errors='replace')
            except Exception:
                key_text = entry.get('key')
            if isinstance(key_text, str) and ':' in key_text:
                worker_id = key_text.rsplit(':', 1)[0]
            else:
                worker_id = str(key_text)
        stats = worker_stats.setdefault(worker_id, {
            'records': 0,
            'iterations': [],
            'queue_sizes': [],
            'car_queue_sizes': [],
            'total_seconds': [],
            'car_seconds': [],
            'can_seconds': [],
            'car_results': defaultdict(int),
            'can_results': defaultdict(int),
            'outcomes': defaultdict(int),
            'events': [],
            'raw_info_totals': defaultdict(float),
            'extensions_totals': defaultdict(float),
        })
        stats['records'] += 1
        iteration = payload.get('iteration')
        if isinstance(iteration, (int, float)):
            stats['iterations'].append(int(iteration))
        queue_size = payload.get('queue_size')
        if isinstance(queue_size, (int, float)):
            stats['queue_sizes'].append(float(queue_size))
        car_queue_size = payload.get('car_queue_size')
        if isinstance(car_queue_size, (int, float)):
            stats['car_queue_sizes'].append(float(car_queue_size))
        timings = payload.get('timings') or {}
        total_seconds = timings.get('total_seconds')
        if isinstance(total_seconds, (int, float)):
            stats['total_seconds'].append(float(total_seconds))
        car_seconds = timings.get('car_seconds')
        if isinstance(car_seconds, (int, float)):
            stats['car_seconds'].append(float(car_seconds))
        can_seconds = timings.get('can_seconds')
        if isinstance(can_seconds, (int, float)):
            stats['can_seconds'].append(float(can_seconds))
        car_result = (payload.get('car_processing') or {}).get('result')
        if car_result:
            stats['car_results'][car_result] += 1
        can_result = (payload.get('can_processing') or {}).get('result')
        if can_result:
            stats['can_results'][can_result] += 1
        for outcome_key, outcome_value in (payload.get('outcomes') or {}).items():
            if isinstance(outcome_value, bool):
                label = f"{outcome_key}={'T' if outcome_value else 'F'}"
                stats['outcomes'][label] += 1
        for prefix, processing in (( 'car', payload.get('car_processing') or {}), ('can', payload.get('can_processing') or {})):
            raw_info = processing.get('raw_info')
            if isinstance(raw_info, dict):
                for key, value in raw_info.items():
                    if isinstance(value, (int, float)):
                        stats['raw_info_totals'][f'{prefix}_{key}'] += float(value)
            extensions = processing.get('extensions')
            if isinstance(extensions, dict):
                for key, value in extensions.items():
                    if isinstance(value, (int, float)):
                        stats['extensions_totals'][f'{prefix}_{key}'] += float(value)
        stats['events'].append(payload)


    def summarize(values):
        if not values:
            return (None, None, None)
        return (min(values), statistics.mean(values), max(values))

    def format_number(value, digits=1):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else '-'
    def format_hours(values):
        total = sum(values)
        return total / 3600 if total else 0.0
    def format_counter(mapping):
        if not mapping:
            return '-'
        items = sorted(dict(mapping).items())
        return ', '.join(f"{key}:{value}" for key, value in items)
    workers = []
    for worker_id, stats in worker_stats.items():
        queue_min, queue_avg, queue_max = summarize(stats['queue_sizes'])
        car_min, car_avg, car_max = summarize(stats['car_queue_sizes'])
        workers.append({
            'worker_id': worker_id,
            'records': stats['records'],
            'iter_min': min(stats['iterations']) if stats['iterations'] else None,
            'iter_max': max(stats['iterations']) if stats['iterations'] else None,
            'queue_avg': queue_avg,
            'car_queue_avg': car_avg,
            'queue_range': (queue_min, queue_max),
            'car_queue_range': (car_min, car_max),
            'total_hours': format_hours(stats['total_seconds']),
            'car_hours': format_hours(stats['car_seconds']),
            'can_hours': format_hours(stats['can_seconds']),
            'car_results': dict(stats['car_results']),
            'can_results': dict(stats['can_results']),
            'outcomes': dict(stats['outcomes']),
        })
    workers.sort(key=lambda item: item['worker_id'])
    return {
        'zip_name': selected_zip_name,
        'db10_file': db10_file,
        'entries': entries,
        'db10_data': db10_data,
        'worker_stats': worker_stats,
        'worker_summaries': workers,
    }


def build_db10_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix, max_events=None, sort_by='iteration'):
    def safe_mean(values):
        return statistics.mean(values) if values else None

    def hours(values):
        return sum(values) / 3600 if values else 0.0

    def event_key(event):
        if sort_by == 'timestamp':
            return (event.get('timestamp_start') or '', event.get('iteration') or float('inf'))
        iteration = event.get('iteration')
        return (iteration if iteration is not None else float('inf'), event.get('timestamp_start') or '')

    def serialize_event(event):
        timings = event.get('timings') or {}
        car_processing = event.get('car_processing') or {}
        can_processing = event.get('can_processing') or {}
        return {
            'iteration': event.get('iteration'),
            'timestamp_start': event.get('timestamp_start'),
            'timestamp_end': event.get('timestamp_end'),
            'queue_size': event.get('queue_size'),
            'car_queue_size': event.get('car_queue_size'),
            'timings': {
                'total_seconds': timings.get('total_seconds'),
                'car_seconds': timings.get('car_seconds'),
                'can_seconds': timings.get('can_seconds'),
            },
            'car_processing': {
                'result': car_processing.get('result'),
                'time_seconds': car_processing.get('time_seconds'),
                'raw_info': car_processing.get('raw_info'),
                'extensions': car_processing.get('extensions'),
            },
            'can_processing': {
                'result': can_processing.get('result'),
                'time_seconds': can_processing.get('time_seconds'),
                'raw_info': can_processing.get('raw_info'),
                'extensions': can_processing.get('extensions'),
            },
            'outcomes': event.get('outcomes'),
        }

    results = summarise_db10_workers(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix)

    worker_stats = results.get("worker_stats")
    report = {
        'worker_stats' : worker_stats,
        'zip_name': results.get('zip_name'),
        'worker_count': len(worker_stats),
        'workers': {}
    }

    for worker_id in sorted(worker_stats):
        stats = worker_stats[worker_id]
        iterations = stats.get('iterations') or []
        queue_sizes = stats.get('queue_sizes') or []
        car_queue_sizes = stats.get('car_queue_sizes') or []
        summary = {
            'records': stats.get('records', 0),
            'iteration_range': [min(iterations), max(iterations)] if iterations else None,
            'queue': {
                'min': min(queue_sizes) if queue_sizes else None,
                'mean': safe_mean(queue_sizes),
                'max': max(queue_sizes) if queue_sizes else None,
            },
            'car_queue': {
                'min': min(car_queue_sizes) if car_queue_sizes else None,
                'mean': safe_mean(car_queue_sizes),
                'max': max(car_queue_sizes) if car_queue_sizes else None,
            },
            'timings_hours': {
                'total': hours(stats.get('total_seconds') or []),
                'car': hours(stats.get('car_seconds') or []),
                'can': hours(stats.get('can_seconds') or []),
            },
            'car_results': dict(stats.get('car_results') or {}),
            'can_results': dict(stats.get('can_results') or {}),
            'outcomes': dict(stats.get('outcomes') or {}),
            'raw_info_totals': dict(stats.get('raw_info_totals') or {}),
            'extensions_totals': dict(stats.get('extensions_totals') or {}),
        }
        events = list(stats.get('events') or [])
        events_sorted = sorted(events, key=event_key)
        if max_events is not None and max_events > 0:
            events_sorted = events_sorted[-max_events:]
        summary['events'] = [serialize_event(event) for event in events_sorted]
        report['workers'][worker_id] = summary

    print(f"Report built for {report['worker_count']} workers.")
    return report

def _build_worker_label_map(worker_ids):
    worker_ids = sorted(worker_ids)
    return {worker_id: f"W{index:02d}" for index, worker_id in enumerate(worker_ids, start=1)}


def _flatten_worker_summary(report):
    rows = []
    car_keys = set()
    can_keys = set()
    outcome_keys = set()
    raw_info_keys = set()
    extensions_keys = set()
    workers = report.get('workers') or {}
    for summary in workers.values():
        car_keys.update((summary.get('car_results') or {}).keys())
        can_keys.update((summary.get('can_results') or {}).keys())
        outcome_keys.update((summary.get('outcomes') or {}).keys())
        raw_info_keys.update((summary.get('raw_info_totals') or {}).keys())
        extensions_keys.update((summary.get('extensions_totals') or {}).keys())
    car_keys = sorted(car_keys)
    can_keys = sorted(can_keys)
    outcome_keys = sorted(outcome_keys)
    raw_info_keys = sorted(raw_info_keys)
    extensions_keys = sorted(extensions_keys)
    sorted_workers = sorted(workers.items())
    for index, (worker_id, summary) in enumerate(sorted_workers, start=1):
        queue = summary.get('queue') or {}
        car_queue = summary.get('car_queue') or {}
        timings = summary.get('timings_hours') or {}
        iteration_range = summary.get('iteration_range') or [None, None]
        short_name = worker_id.split(':')[-1]
        label = f"W{index:02d}"
        row = OrderedDict(
            worker_id=worker_id,
            worker_index=index,
            worker_label=label,
            worker_short_name=short_name,
            records=summary.get('records'),
            iter_min=iteration_range[0],
            iter_max=iteration_range[1],
            queue_min=queue.get('min'),
            queue_mean=queue.get('mean'),
            queue_max=queue.get('max'),
            car_queue_min=car_queue.get('min'),
            car_queue_mean=car_queue.get('mean'),
            car_queue_max=car_queue.get('max'),
            hours_total=timings.get('total'),
            hours_car=timings.get('car'),
            hours_can=timings.get('can'),
        )
        car_results = summary.get('car_results') or {}
        can_results = summary.get('can_results') or {}
        outcomes = summary.get('outcomes') or {}
        for key in car_keys:
            row[f'car_result_{key}'] = car_results.get(key, 0)
        for key in can_keys:
            row[f'can_result_{key}'] = can_results.get(key, 0)
        for key in outcome_keys:
            row[f'outcome_{key}'] = outcomes.get(key, 0)
        raw_info_totals = summary.get('raw_info_totals') or {}
        extensions_totals = summary.get('extensions_totals') or {}
        for key in raw_info_keys:
            value = raw_info_totals.get(key, 0.0)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            row[f'raw_info_{key}'] = value
        for key in extensions_keys:
            value = extensions_totals.get(key, 0.0)
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            row[f'extensions_{key}'] = value
        rows.append(row)
    return rows


def _token_to_display(token: str) -> str:
    upper = token.upper()
    if upper in DB_DISPLAY_NAMES:
        return DB_DISPLAY_NAMES[upper]
    lookup = {
        'CAR': CANDIDATE_ANTI_REASONS_NAME,
        'CAN': CANDIDATE_REASONS_NAME,
    }
    if upper in lookup:
        return lookup[upper]
    if token.isupper():
        return token.capitalize()
    return token.replace('_', ' ').capitalize()


def _format_result_label(prefix: str, raw: str) -> str:
    parts = raw.split('_') if raw else []
    if parts and parts[-1].upper() in DB_DISPLAY_NAMES:
        parts = parts[:-1]
    descriptor = ' '.join(_token_to_display(part.lower()) for part in parts) if parts else 'Total'
    return f"{prefix} result: {descriptor}"


def _format_outcome_label(raw: str) -> str:
    key, sep, value = raw.partition('=')
    tokens = key.split('_') if key else []
    words = [_token_to_display(tok) for tok in tokens]
    label = ' '.join(words) if words else 'Outcome'
    if sep:
        value_text = {'T': 'True', 'F': 'False', 'TRUE': 'True', 'FALSE': 'False'}.get(value.upper(), value)
        return f"Outcome: {label} ({value_text})"
    return f"Outcome: {label}"


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        'worker_label': 'Worker',
        'records': 'Records',
        'iter_min': 'Iteration min',
        'iter_max': 'Iteration max',
        'queue_min': 'Queue min',
        'queue_mean': 'Queue mean',
        'queue_max': 'Queue max',
        'car_queue_min': f"{CANDIDATE_ANTI_REASONS_NAME} queue min",
        'car_queue_mean': f"{CANDIDATE_ANTI_REASONS_NAME} queue mean",
        'car_queue_max': f"{CANDIDATE_ANTI_REASONS_NAME} queue max",
        'hours_total': 'Total hours',
        'hours_car': f"{CANDIDATE_ANTI_REASONS_NAME} hours",
        'hours_can': f"{CANDIDATE_REASONS_NAME} hours",
        'event_order': 'Event order',
        'timestamp_start': 'Start time',
        'timestamp_end': 'End time',
        'car_queue_size': f"{CANDIDATE_ANTI_REASONS_NAME} queue size",
        'total_seconds': 'Total seconds',
        'car_seconds': f"{CANDIDATE_ANTI_REASONS_NAME} seconds",
        'can_seconds': f"{CANDIDATE_REASONS_NAME} seconds",
        'car_result': f"{CANDIDATE_ANTI_REASONS_NAME} result",
        'can_result': f"{CANDIDATE_REASONS_NAME} result",
        'outcomes': 'Outcomes',
    }
    def _prettify_metric(name: str) -> str:
        parts = name.replace('-', '_').split('_')
        return ' '.join(part.upper() if len(part) <= 3 and part.isupper() else part.capitalize() for part in parts if part)
    def _format_metric(metric_key: str, category: str) -> str:
        prefix_map = {
            'car': CANDIDATE_ANTI_REASONS_NAME,
            'can': CANDIDATE_REASONS_NAME,
        }
        if '_' in metric_key:
            prefix, remainder = metric_key.split('_', 1)
            if prefix in prefix_map:
                return f"{prefix_map[prefix]} {category.lower()} {_prettify_metric(remainder)}"
        return f"{category} {_prettify_metric(metric_key)}"
    computed = {}
    for col in df.columns:
        if col.startswith('car_result_'):
            computed[col] = _format_result_label(CANDIDATE_ANTI_REASONS_NAME, col[len('car_result_'):])
        elif col.startswith('can_result_'):
            computed[col] = _format_result_label(CANDIDATE_REASONS_NAME, col[len('can_result_'):])
        elif col.startswith('outcome_'):
            computed[col] = _format_outcome_label(col[len('outcome_'):])
        elif col.startswith('raw_info_'):
            metric = col[len('raw_info_'):]
            computed[col] = _format_metric(metric, 'Raw info')
        elif col.startswith('extensions_'):
            metric = col[len('extensions_'):]
            computed[col] = _format_metric(metric, 'Extensions')
        elif col.startswith('car_raw_info_'):
            metric = f"car_{col[len('car_raw_info_'):]}"
            computed[col] = _format_metric(metric, 'Raw info')
        elif col.startswith('can_raw_info_'):
            metric = f"can_{col[len('can_raw_info_'):]}"
            computed[col] = _format_metric(metric, 'Raw info')
        elif col.startswith('car_extensions_'):
            metric = f"car_{col[len('car_extensions_'):]}"
            computed[col] = _format_metric(metric, 'Extensions')
        elif col.startswith('can_extensions_'):
            metric = f"can_{col[len('can_extensions_'):]}"
            computed[col] = _format_metric(metric, 'Extensions')
        elif col in rename_map:
            computed[col] = rename_map[col]
        else:
            computed[col] = _prettify_metric(col)
    return df.rename(columns=computed)


def _plot_scatter(df: pd.DataFrame, config: dict):
    points = _filter_valid_points(df, config['x'], config['y'])
    if not points:
        print(f"Plot '{config['title']}' skipped: insufficient data.")
        return
    plt.figure(figsize=(8, 5))
    xs, ys, labels = zip(*points)
    plt.scatter(xs, ys, alpha=0.7)
    for x, y, label in points:
        plt.annotate(label, (x, y), textcoords='offset points', xytext=(5, 3), fontsize=8)
    xlabel = config.get('xlabel') or _column_display_name(config['x'])
    ylabel = config.get('ylabel') or _column_display_name(config['y'])
    plt.title(config['title'])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def _plot_bar(df: pd.DataFrame, *, columns: Iterable[str], title: str, ylabel: str, stacked: bool = False, sort_by: str | None = None):
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        raise ValueError(f"Plot '{title}' skipped: columns not available.")
    subset = df[['worker_label'] + available_columns].copy()
    subset = subset.dropna(how='all', subset=available_columns)
    if subset.empty:
        raise ValueError(f"Plot '{title}' skipped: insufficient data.")

    subset[available_columns] = subset[available_columns].apply(pd.to_numeric, errors='coerce')
    subset = subset.dropna()
    if subset.empty:
        raise ValueError(f"Plot '{title}' skipped: non-numeric data.")

    renamed_subset = _rename_columns(subset)
    plot_df = renamed_subset.set_index('Worker')
    value_columns = []
    for original in available_columns:
        display_name = _column_display_name(original)
        if display_name in plot_df.columns:
            value_columns.append(display_name)
    if not value_columns:
        raise ValueError(f"Plot '{title}' skipped: no numeric columns available after renaming.")

    sort_column = _column_display_name(sort_by) if sort_by else None
    if sort_column and sort_column in plot_df.columns:
        plot_df = plot_df.sort_values(sort_column, ascending=False)
    plt.figure(figsize=(max(8, len(plot_df) * 0.5), 5))
    plot_df[value_columns].plot(kind='bar', stacked=stacked, alpha=0.8)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Worker')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    return plt


def _collect_heatmap_rows(entries):
    extracted = []
    for entry in entries or []:
        try:
            key_bytes = decode_key(entry)
            key_text = key_bytes.decode('utf-8', errors='replace')
        except Exception:
            continue
        preview, details = try_decode_value(entry)
        timestamp = None
        if isinstance(preview, bytes):
            preview = preview.decode('utf-8', errors='replace')
        if isinstance(preview, str):
            candidate = preview.strip()
            if candidate:
                timestamp = candidate
        if timestamp is None and isinstance(details, dict):
            decoded_bytes = details.get('decoded_bytes')
            if isinstance(decoded_bytes, (bytes, bytearray)):
                candidate = decoded_bytes.decode('utf-8', errors='replace').strip()
                if candidate:
                    timestamp = candidate
        extracted.append((timestamp, key_text))
    if not extracted:
        return np.empty((0, 0), dtype=int), []
    def _sort_key(item):
        ts, _ = item
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return datetime.max
    extracted.sort(key=_sort_key)
    width = max(len(key) for _, key in extracted)
    matrix = np.full((len(extracted), width), np.nan, dtype=float)
    labels = []
    for row_idx, (ts, key_text) in enumerate(extracted):
        if len(key_text) != width:
            key_text = key_text.ljust(width, '0')
        matrix[row_idx, :] = [1 if ch == '1' else 0 for ch in key_text]
        labels.append(ts or 'n/a')
    return matrix, labels


def _collect_worker_iteration_rows(report, df: pd.DataFrame):
    worker_stats = report.get('worker_stats') or {}
    workers = (report or {}).get('workers') or {}
    union_worker_ids = set(workers.keys()) | set(worker_stats.keys())
    label_map = _build_worker_label_map(union_worker_ids)
    rows = []
    for worker_id in sorted(union_worker_ids):
        summary = workers.get(worker_id) or {}
        events = list(summary.get('events') or [])
        if not events and worker_stats:
            events = list((worker_stats.get(worker_id) or {}).get('events') or [])
        if not events:
            continue
        events_sorted = sorted(events, key=_queue_event_sort_key)
        label = label_map.get(worker_id)
        for order, event in enumerate(events_sorted, start=1):
            timings = event.get('timings') or {}
            car_processing = event.get('car_processing') or {}
            can_processing = event.get('can_processing') or {}
            outcomes = event.get('outcomes')
            if isinstance(outcomes, dict):
                outcome_text = ', '.join(
                    f"{key}={'T' if value else 'F'}" for key, value in sorted(outcomes.items())
                )
            else:
                outcome_text = None
            row = {
                'worker_id': worker_id,
                'worker_label': label,
                'event_order': order,
                'iteration': event.get('iteration'),
                'timestamp_start': event.get('timestamp_start'),
                'timestamp_end': event.get('timestamp_end'),
                'queue_size': _extract_numeric_value(event.get('queue_size'), event.get('queue')),
                'car_queue_size': _extract_numeric_value(event.get('car_queue_size'), event.get('car_queue')),
                'total_seconds': timings.get('total_seconds'),
                'car_seconds': timings.get('car_seconds'),
                'can_seconds': timings.get('can_seconds'),
                'car_result': car_processing.get('result'),
                'can_result': can_processing.get('result'),
                'outcomes': outcome_text,
            }
            for prefix, processing in (( 'car', car_processing), ('can', can_processing)):
                raw_info = processing.get('raw_info')
                if isinstance(raw_info, dict):
                    for key, value in raw_info.items():
                        numeric = _extract_numeric_value(value)
                        if numeric is None:
                            continue
                        if isinstance(numeric, float) and numeric.is_integer():
                            numeric = int(numeric)
                        row[f"{prefix}_raw_info_{key}"] = numeric
                extensions = processing.get('extensions')
                if isinstance(extensions, dict):
                    for key, value in extensions.items():
                        numeric = _extract_numeric_value(value)
                        if numeric is None:
                            continue
                        if isinstance(numeric, float) and numeric.is_integer():
                            numeric = int(numeric)
                        row[f"{prefix}_extensions_{key}"] = numeric
            rows.append(row)
    return rows


def _filter_valid_points(df: pd.DataFrame, x_key: str, y_key: str):
    subset = df.dropna(subset=[x_key, y_key])
    subset = subset[
        subset[[x_key, y_key]].map(lambda v: isinstance(v, (int, float))).all(axis=1)
    ]
    if subset.empty:
        return []
    return list(zip(subset[x_key], subset[y_key], subset['worker_label']))


def _column_display_name(column: str) -> str:
    temp = _rename_columns(pd.DataFrame(columns=[column]))
    return temp.columns[0]


def _queue_event_sort_key(event):
    iteration = event.get('iteration')
    timestamp = event.get('timestamp_start') or event.get('start') or ''
    return (
        iteration if isinstance(iteration, (int, float)) else float('inf'),
        timestamp,
    )


def _extract_numeric_value(*candidates):
    for value in candidates:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def render_worker_report(db):
    report = db.get(DB_NAMES.get(10), None)

    if report is None:
        raise ValueError('No worker report (DB 10) available in the database, Run etl.')

    rows = _flatten_worker_summary(report)
    if not rows:
        raise ValueError('No data available for the worker report.')

    df = pd.DataFrame(rows)
    df = df.sort_values('worker_index')
    base_columns = [col for col in BASE_COLUMNS if col in df.columns]
    extra_columns = sorted(
        col for col in df.columns if col.startswith('car_result_') or col.startswith('can_result_') or col.startswith('outcome_')
    )
    base_table_columns = ['worker_label'] + [col for col in BASE_COLUMNS if col not in {'worker_id'}]
    base_table_columns = [col for col in base_table_columns if col in df.columns]

    workers_table = _rename_columns(df[base_table_columns])

    print('Detailed results per worker:')
    raw_info_columns = sorted(col for col in df.columns if col.startswith('raw_info_'))
    extensions_columns = sorted(col for col in df.columns if col.startswith('extensions_'))
    detail_columns = ['worker_label'] + extra_columns + raw_info_columns + extensions_columns
    detail_columns = [col for col in detail_columns if col in df.columns]

    workers_table2 = _rename_columns(df[detail_columns])
    plots = []
    for config in SCATTER_PLOTS:
        plots.append(_plot_scatter(df, config))
    for x_key, y_key, title in ADDITIONAL_SCATTER_PREFIX_PAIRS:
        if x_key in df.columns and y_key in df.columns:
            plots.append(_plot_scatter(
                df,
                {
                    'title': title,
                    'x': x_key,
                    'y': y_key,
                    'xlabel': _column_display_name(x_key),
                    'ylabel': _column_display_name(y_key),
                },
            ))
    for config in BAR_PLOTS:
        plots.append(_plot_bar(
            df,
            columns=config['columns'],
            title=config['title'],
            ylabel=config['ylabel'],
            stacked=False,
            sort_by=config.get('sort_by'),
        ))
    for config in STACKED_BAR_CONFIG:
        columns = sorted(col for col in df.columns if col.startswith(config['prefix']))
        if columns:
            plots.append(_plot_bar(
                df,
                columns=columns,
                title=config['title'],
                ylabel=config['ylabel'],
                stacked=True,
            ))
        else:
            print(f"Plot '{config['title']}' skipped: no columns with prefix {config['prefix']!r}.")

    for config in HISTOGRAMS:
        plots.append(_plot_histogram(df, column=config['column'], title=config['title'], xlabel=config['xlabel']))
    if extensions_columns:
        plots.append(_plot_bar(df, columns=extensions_columns, title='Extensions totals per worker', ylabel='Count', stacked=False))
    plots.extend(_plot_queue_time_series(report, df))
    workers_table3 = _render_worker_iteration_table(report, df)

    return workers_table, workers_table2, workers_table3, plots


def _plot_queue_time_series(report, df: pd.DataFrame):
    workers_data = (report or {}).get('workers') or {}
    worker_stats = report.get("worker_stats")
    if not workers_data and not worker_stats:
        raise ValueError("Queue time series skipped: worker events unavailable.")

    union_worker_ids = set(workers_data.keys()) | set(worker_stats.keys())
    label_map = _build_worker_label_map(union_worker_ids)
    queue_series = []
    car_queue_series = []
    for worker_id in sorted(union_worker_ids):
        summary = workers_data.get(worker_id, {})
        events = list(summary.get('events') or [])
        if not events and worker_stats:
            events = list((worker_stats.get(worker_id) or {}).get('events') or [])
        if not events:
            continue
        events_sorted = sorted(events, key=_queue_event_sort_key)
        queue_points = []
        car_points = []
        label = label_map.get(worker_id)
        for order, event in enumerate(events_sorted):
            timestamp = _parse_worker_event_timestamp(event)
            iteration = event.get('iteration')
            iteration_value = int(iteration) if isinstance(iteration, (int, float)) else None
            queue_value = _extract_numeric_value(event.get('queue_size'), event.get('queue'))
            if queue_value is not None:
                queue_points.append({
                    'timestamp': timestamp,
                    'iteration': iteration_value,
                    'order': order + 1,
                    'value': queue_value,
                })
            car_queue_value = _extract_numeric_value(event.get('car_queue_size'), event.get('car_queue'))
            if car_queue_value is not None:
                car_points.append({
                    'timestamp': timestamp,
                    'iteration': iteration_value,
                    'order': order + 1,
                    'value': car_queue_value,
                })
        if queue_points:
            queue_series.append((label, queue_points))
        if car_points:
            car_queue_series.append((label, car_points))

    return [p for p in [
        _render_combined_timeseries(queue_series, title='Queue size overview', ylabel='Queue size'),
        _render_combined_timeseries(car_queue_series, title=f"{CANDIDATE_ANTI_REASONS_NAME} queue overview", ylabel=f"{CANDIDATE_ANTI_REASONS_NAME} queue size")
    ] if p is not None]

def _render_worker_iteration_table(report, df: pd.DataFrame, *, max_rows=5000):
    rows = _collect_worker_iteration_rows(report, df)
    if not rows:
        print('Worker iteration details are not available for the current selection.')
        return pd.DataFrame()

    table = pd.DataFrame(rows)
    if table.empty:
        print('Worker iteration details are not available for the current selection.')
        return table

    for column in ('timestamp_start', 'timestamp_end'):
        if column in table.columns:
            table[column] = pd.to_datetime(table[column], errors='coerce')

    sort_columns = ['worker_label', 'event_order']
    if 'timestamp_start' in table.columns and table['timestamp_start'].notna().any():
        sort_columns = ['worker_label', 'timestamp_start', 'event_order']
    elif 'iteration' in table.columns and table['iteration'].notna().any():
        sort_columns = ['worker_label', 'iteration', 'event_order']
    table = table.sort_values(sort_columns)
    total_rows = len(table)
    worker_count = table['worker_label'].nunique() if 'worker_label' in table.columns else None
    if max_rows is not None and total_rows > max_rows:
        summary = f"Showing first {max_rows} of {total_rows} worker iterations"
        if worker_count is not None:
            summary += f" across {worker_count} workers"
        print(summary + '.')
        display_df = table.head(max_rows)
    else:
        summary = f"Worker iterations: {total_rows} rows"
        if worker_count is not None:
            summary += f" across {worker_count} workers"
        print(summary + '.')
        display_df = table
    if 'worker_id' in display_df.columns:
        display_df = display_df.drop(columns=['worker_id'])
    return _rename_columns(display_df)


def _plot_histogram(df: pd.DataFrame, *, column: str, title: str, xlabel: str):
    if column not in df.columns:
        print(f"Histogram '{title}' skipped: column not available.")
        return
    series = pd.to_numeric(df[column], errors='coerce').dropna()
    if series.empty:
        print(f"Histogram '{title}' skipped: insufficient data.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(series, bins=min(25, len(series)), alpha=0.7)
    label = xlabel or _column_display_name(column)
    plt.title(title)
    plt.xlabel(label)
    plt.ylabel('Number of workers')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def _parse_worker_event_timestamp(event):
    candidates = [
        event.get('timestamp_start'),
        event.get('start'),
        event.get('started_at'),
        event.get('start_time'),
        event.get('timestamp'),
    ]
    for ts in candidates:
        if isinstance(ts, str) and ts:
            parsed = pd.to_datetime(ts, utc=False, errors='coerce')
            if pd.notna(parsed):
                return parsed.to_pydatetime()
    return None


def _draw_worker_timeseries(ax, items, ylabel: str):
    timestamp_points = [(entry['timestamp'], entry['value']) for entry in items if entry['timestamp'] is not None]
    if timestamp_points:
        xs = []
        for ts, _ in timestamp_points:
            if hasattr(ts, 'to_pydatetime'):
                xs.append(ts.to_pydatetime())
            else:
                xs.append(ts)
        ys = [value for _, value in timestamp_points]
        ax.plot(xs, ys, marker='o', linewidth=1)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.set_xlabel('Start time')
    else:
        iteration_points = [(entry['iteration'], entry['value']) for entry in items if entry['iteration'] is not None]
        if iteration_points:
            xs = [float(it) for it, _ in iteration_points]
            ys = [value for _, value in iteration_points]
            ax.plot(xs, ys, marker='o', linewidth=1)
            ax.set_xlabel('Iteration')
        else:
            xs = [entry['order'] for entry in items]
            ys = [entry['value'] for entry in items]
            ax.plot(xs, ys, marker='o', linewidth=1)
            ax.set_xlabel('Event order')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def _render_worker_timeseries_grid(series_data, *, title: str, ylabel: str):
    if not series_data:
        print(f"Plot '{title}' skipped: insufficient data.")
        return
    if widgets is None:
        cols = 2
        rows = (len(series_data) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3), squeeze=False)
        flat_axes = axes.flatten()
        for ax in flat_axes[len(series_data):]:
            ax.axis('off')
        for ax, (worker_label, items) in zip(flat_axes, series_data):
            _draw_worker_timeseries(ax, items, ylabel)
            ax.set_title(worker_label, fontsize=9)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()
        return

    series_map = {label: items for label, items in series_data}
    labels = [label for label, _ in series_data]

    def _render_single_worker(worker_label: str):
        fig, ax = plt.subplots(figsize=(10, 4))
        _draw_worker_timeseries(ax, series_map[worker_label], ylabel)
        ax.set_title(worker_label, fontsize=10)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        plt.show()

    if len(labels) == 1:
        _render_single_worker(labels[0])
        return

    dropdown = widgets.Dropdown(options=labels, value=labels[0], description='Worker')
    output = widgets.Output()

    def _update_plot(label):
        if label is None:
            return
        output.clear_output(wait=True)
        with output:
            _render_single_worker(label)

    def _on_change(change):
        if change['name'] == 'value':
            _update_plot(change['new'])

    dropdown.observe(_on_change, names='value')
    _update_plot(labels[0])
    display(widgets.VBox([widgets.HTML(f"<b>{title}</b>"), dropdown, output]))


def _render_combined_timeseries(series_data, *, title: str, ylabel: str):
    if not series_data:
        print(f"Plot '{title}' skipped: insufficient data.")
        return None

    plt.figure(figsize=(12, 4))
    for label, items in series_data:
        timestamps = [entry['timestamp'] for entry in items if entry['timestamp'] is not None]
        if timestamps:
            xs = []
            for ts in timestamps:
                if hasattr(ts, 'to_pydatetime'):
                    xs.append(ts.to_pydatetime())
                else:
                    xs.append(ts)
            ys = [entry['value'] for entry in items if entry['timestamp'] is not None]
            plt.plot(xs, ys, label=label, linewidth=1)
        else:
            iterations = [entry['iteration'] for entry in items if entry['iteration'] is not None]
            if iterations:
                xs = iterations
                ys = [entry['value'] for entry in items if entry['iteration'] is not None]
                plt.plot(xs, ys, label=label, linewidth=1)
            else:
                xs = [entry['order'] for entry in items]
                ys = [entry['value'] for entry in items]
                plt.plot(xs, ys, label=label, linewidth=1)
    plt.title(title)
    plt.ylabel(ylabel)
    has_time = any(entry['timestamp'] is not None for _, items in series_data for entry in items)
    plt.xlabel('Time' if has_time else 'Iteration/order')
    if len(series_data) <= 12:
        plt.legend(fontsize=7)
    else:
        plt.legend(fontsize=6, ncol=2, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt

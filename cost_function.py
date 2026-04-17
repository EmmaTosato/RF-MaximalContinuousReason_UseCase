from random import random
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy.stats import norm

EPS = np.finfo(float).tiny
MIN_SIGMA = 1e-10  # Minimum meaningful sigma value to prevent numerical issues
HALF = 0.5


def cal_sigmas(X_train, X_test, feature_names, test_ids=None):
	"""
	Calculate sigma_plus and sigma_minus for each feature in X_test based on X_train.
	If test_ids is provided, the outer dictionary will use those IDs instead of numeric indices.

	Parameters
	----------
	X_train : array-like
		Training samples (n_train, n_features)
	X_test : array-like
		Test samples (n_test, n_features)
	feature_names : list of str
		Names of the features (columns)
	test_ids : list, optional
		Custom identifiers for each row of X_test (e.g., keys of tests_sample).
		If None, the function defaults to numeric indices 0..n_test-1.

	Returns
	-------
	sigmas_all : dict
		Dictionary keyed by either numeric indices or test_ids.
		Each entry maps feature_name -> dict with:
			sigma_plus, sigma_minus, ratio_above_mean, ratio_below_mean
	"""

	sigmas_all = {}
	X_train_df = pd.DataFrame(X_train, columns=feature_names)
	X_test_df = pd.DataFrame(X_test, columns=feature_names)

	# Default behaviour preserved if test_ids not given
	if test_ids is None:
		test_ids = list(range(len(X_test_df)))

	# Sanity check
	if len(test_ids) != len(X_test_df):
		raise ValueError(
			f"Length mismatch: len(test_ids)={len(test_ids)} vs len(X_test)={len(X_test_df)}"
		)

	# Main computation
	for sample_id, (_, row) in zip(test_ids, X_test_df.iterrows()):
		sigmas_all[sample_id] = {}
		for feature in feature_names:
			tmp = np.array(X_train_df[feature]) - row[feature]
			delta_pos = tmp[tmp >= 0]
			delta_neg = np.abs(tmp[tmp < 0])

			n_above = np.float64(delta_pos.shape[0])
			n_below = np.float64(delta_neg.shape[0])
			n = n_above + n_below

			if n == 0:
				continue

			sum_pos = np.sum(delta_pos ** 2)
			sum_neg = np.sum(delta_neg ** 2)

			sigma_plus = float(np.sqrt(sum_pos / n_above)) if n_above > 0 else 0.0
			sigma_minus = float(np.sqrt(sum_neg / n_below)) if n_below > 0 else 0.0

			sigmas_all[sample_id][feature] = {
				"sigma_plus": sigma_plus,
				"sigma_minus": sigma_minus,
				"ratio_above_mean": float(n_above / n) if n > 0 else 0.0,
				"ratio_below_mean": float(n_below / n) if n > 0 else 0.0,
			}

	return sigmas_all

def cost_function(sample: Dict[str, float] = None,  icf: Dict[str, Tuple[float, float]] = None, sigmas: Dict[str, Dict[str, dict]] = None, calculate_missing_features_in_icf: bool = False, verbose: bool = False, plot: bool = False) -> float:
	"""
	Calculate cost function based on split Gaussian distributions.

	Parameters
	----------
	sample : dict
		Sample values for each feature
	icf : dict
		Interval for each feature (min, max)
	sigmas : dict
		Sigma values and ratios for each feature
	calculate_missing_features_in_icf : bool
		Whether to calculate cost for missing features in ICF
	verbose : bool
		Print debug information

	Returns
	-------
	float
		Total cost across all features
	"""
	if sigmas is None:
		raise ValueError("Sigmas must be provided")
	if icf is None:
		raise ValueError("ICF must be provided")
	if sample is None:
		raise ValueError("Sample must be provided")
	
	cost = 0.0
	# plots_done is deprecated; we now plot every feature when verbose is True

	for key in icf.keys():
		if verbose:
			print(f"Processing key: {key}")

		# Skip if key not in sigmas (feature might have been skipped in cal_sigmas)
		if key not in sigmas:
			if verbose:
				print(f"  Warning: key {key} not in sigmas, skipping")
			continue

		sigma_pos = sigmas[key]['sigma_plus']
		sigma_neg = sigmas[key]['sigma_minus']
		percent_above = sigmas[key]['ratio_above_mean']
		percent_below = sigmas[key]['ratio_below_mean']
		interval_min, interval_max = icf[key]

		if verbose:
			print(f"  Interval: [{interval_min:.4f}, {interval_max:.4f}]")
			print(f"  Sigmas: sigma_pos={sigma_pos:.4f}, sigma_neg={sigma_neg:.4f}")
			print(f"  Percentages: above={percent_above:.4f}, below={percent_below:.4f}")

		# Validate percentages
		if not np.isclose(percent_above + percent_below, 1.0, rtol=1e-5):
			if verbose:
				print(f"  Warning: percentages for key {key} don't sum to 1.0: sum={percent_above + percent_below}")
			# Normalize percentages if close enough
			total_percent = percent_above + percent_below
			if total_percent > 0:
				percent_above /= total_percent
				percent_below /= total_percent
			else:
				# If both are zero, skip this feature
				if verbose:
					print(f"  Skipping feature {key} due to zero percentages")
				continue

		# Protect against zero or negative sigmas
		sigma_neg = max(abs(sigma_neg), MIN_SIGMA)
		sigma_pos = max(abs(sigma_pos), MIN_SIGMA)

		# If both sigmas are at minimum threshold, it means no variance - skip this feature
		if sigma_neg <= MIN_SIGMA and sigma_pos <= MIN_SIGMA:
			if verbose:
				print(f"  Skipping feature {key} due to zero variance (both sigmas at minimum)")
			continue

		# Shift interval to be relative to sample value
		# According to paper: interval (b - x[f], e - x[f])
		interval_min_shifted = interval_min - sample[key]
		interval_max_shifted = interval_max - sample[key]

		if verbose:
			print(f"  Interval shifted: [{interval_min_shifted:.4f}, {interval_max_shifted:.4f}]")


		# Calculate cost contribution for this feature
		# According to paper formula:
		# A_{x,f}(b,e) = p_{x,f}^+ * ∫_b^e N(x;0,σ+²)dx + p_{x,f}^- * ∫_b^e N(x;0,σ-²)dx
		# Using CDF: ∫_b^e N(x;0,σ²)dx = Φ(e/σ) - Φ(b/σ)
		# where Φ is the standard normal CDF
		try:
			# Positive contribution: p^+ * [Φ(e/σ+) - Φ(b/σ+)]
			cdf_max_pos = norm.cdf(interval_max_shifted / sigma_pos)
			cdf_min_pos = norm.cdf(interval_min_shifted / sigma_pos)
			area_pos = percent_above * (cdf_max_pos - cdf_min_pos)

			# Negative contribution: p^- * [Φ(e/σ-) - Φ(b/σ-)]
			cdf_max_neg = norm.cdf(interval_max_shifted / sigma_neg)
			cdf_min_neg = norm.cdf(interval_min_shifted / sigma_neg)
			area_neg = percent_below * (cdf_max_neg - cdf_min_neg)

			# Total area for this feature
			area = area_pos + area_neg

			# Sanity check: area should be between 0 and 1
			if not (0 <= area <= 1.0 + 1e-6):
				if verbose:
					print(f"  Warning: area {area:.6f} outside [0,1] for key {key}, clamping")
				area = np.clip(area, 0.0, 1.0)

		except Exception as e:
			if verbose:
				print(f"  Warning: CDF calculation error for interval in key {key}: {e}")
			area = 0.0

		# Cost is area under the curve in the interval
		cost += area
		if calculate_missing_features_in_icf:
			cost += (len(list(sample.keys()))-len(list(icf.keys())) )
		if plot:
			print(f"  Area under curve in interval: {area:.4f}, Cost total: {cost:.4f}")

			# Plot the curve and highlight the interval (now one plot per feature when verbose)
			try:
				# Define split PDF according to paper formula (piecewise, not symmetric)
				def split_pdf(x):
					x = np.asarray(x)
					pdf_pos = percent_above * (1 / (sigma_pos * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_pos) ** 2)
					pdf_neg = percent_below * (1 / (sigma_neg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_neg) ** 2)
					return np.where(x >= 0, pdf_pos, 0.0) + np.where(x < 0, pdf_neg, 0.0)

				# Calculate total areas using CDF for visualization
				# Area from -inf to +inf should equal 1.0
				area_total_pos = percent_above  # ∫_{-∞}^{+∞} p^+ * N(x;0,σ+²)dx = p^+
				area_total_neg = percent_below  # ∫_{-∞}^{+∞} p^- * N(x;0,σ-²)dx = p^-

				print(f"  Area positive Gaussian: {area_total_pos:.4f}, Area negative Gaussian: {area_total_neg:.4f}, Total: {area_total_pos + area_total_neg:.4f}")

				# Use dynamic window like plot_cost_distribution for clarity
				candidates = []
				if not np.isinf(interval_min_shifted):
					candidates.append(interval_min_shifted)
				if not np.isinf(interval_max_shifted):
					candidates.append(interval_max_shifted)
				candidates.extend([-4 * sigma_neg, 4 * sigma_pos, -4 * sigma_pos, 4 * sigma_neg])
				if len(candidates) == 0:
					candidates = [-1.0, 1.0]
				base_min = min(candidates)
				base_max = max(candidates)
				if np.isclose(base_min, base_max):
					base_min -= 1.0
					base_max += 1.0
				padding = max(0.1, 0.1 * (base_max - base_min))
				x_min = base_min - padding
				x_max = base_max + padding

				x_vals = np.linspace(x_min, x_max, 600)
				y_vals = split_pdf(x_vals)
				plt.figure(figsize=(8, 4))
				plt.plot(x_vals, y_vals)
				plt.fill_between(x_vals, 0, y_vals, alpha=0.3, label=f'PDF (p+={percent_above:.2f}, p-={percent_below:.2f})')

				raw_plot_min = interval_min_shifted if not np.isinf(interval_min_shifted) else x_min
				raw_plot_max = interval_max_shifted if not np.isinf(interval_max_shifted) else x_max
				plot_min = max(x_min, raw_plot_min)
				plot_max = min(x_max, raw_plot_max)

				plt.axvspan(plot_min, plot_max, color='black', alpha=0.4, label='Interval')

				# Create title with proper inf handling
				interval_str = f"[{interval_min_shifted:.4f}, {interval_max_shifted:.4f}]"
				if np.isinf(interval_min_shifted):
					interval_str = f"[-∞, {interval_max_shifted:.4f}]"
				if np.isinf(interval_max_shifted):
					interval_str = f"[{interval_min_shifted:.4f}, ∞]"
				if np.isinf(interval_min_shifted) and np.isinf(interval_max_shifted):
					interval_str = "[-∞, ∞]"

				plt.title(f'Feature: {key} | Cost contribution: {area:.4f} | Interval: {interval_str} | Sigmas: +{sigma_pos:.2f}, -{sigma_neg:.2f}')
				plt.axvline(0, color='black', linestyle='--')
				plt.legend()
				plt.savefig(f'fig/feature_{key}_cost_plot.png')
				plt.close()
				print(f"  Saved distribution plot for feature {key} to fig/feature_{key}_cost_plot.png")
			except Exception as e:
				if verbose:
					print(f"  Warning: plotting error for key {key}: {e}")

	return cost

def plot_cost_distribution(sample, icf, sigmas, feature_key, output_dir='fig'):
	"""
	Plot the cost distribution for a specific feature showing the split Gaussian.
	Parameters
	----------
	sample : dict
	Sample values for each feature
	icf : dict
	Interval for each feature (min, max)
	sigmas : dict
	Sigma values and ratios for the feature
	feature_key : str
	The feature to plot
	output_dir : str
	Directory to save plots (default: 'fig')
	"""
	from pathlib import Path
	# Create output directory if needed
	Path(output_dir).mkdir(parents=True, exist_ok=True)
	if feature_key not in sigmas:
		print(f"Warning: feature {feature_key} not in sigmas")
		return
	if feature_key not in icf:
		print(f"Warning: feature {feature_key} not in icf")
		return
	sigma_pos = sigmas[feature_key]['sigma_plus']
	sigma_neg = sigmas[feature_key]['sigma_minus']
	percent_above = sigmas[feature_key]['ratio_above_mean']
	percent_below = sigmas[feature_key]['ratio_below_mean']
	interval_min, interval_max = icf[feature_key]
	# Protect against zero sigmas
	sigma_neg = max(abs(sigma_neg), MIN_SIGMA)
	sigma_pos = max(abs(sigma_pos), MIN_SIGMA)
	# Shift interval
	interval_min_shifted = interval_min - sample[feature_key]
	interval_max_shifted = interval_max - sample[feature_key]
	# Define split PDF
	def split_pdf(x):
		x = np.asarray(x)
		pdf_pos = percent_above * (1 / (sigma_pos * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_pos) ** 2)
		pdf_neg = percent_below * (1 / (sigma_neg * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma_neg) ** 2)
		# Piecewise: use σ+ for x>=0 and σ- for x<0 so the combined curve is asymmetric if weights/sigmas differ
		return np.where(x >= 0, pdf_pos, 0.0) + np.where(x < 0, pdf_neg, 0.0)
	# Calculate area in interval using CDF
	if interval_max_shifted <= 0:
		cdf_max = norm.cdf(interval_max_shifted / sigma_neg)
		cdf_min = norm.cdf(interval_min_shifted / sigma_neg)
		area = percent_below * (cdf_max - cdf_min)
	elif interval_min_shifted >= 0:
		cdf_max = norm.cdf(interval_max_shifted / sigma_pos)
		cdf_min = norm.cdf(interval_min_shifted / sigma_pos)
		area = percent_above * (cdf_max - cdf_min)
	else:
		# Split interval
		cdf_0_neg = norm.cdf(0 / sigma_neg)
		cdf_min_neg = norm.cdf(interval_min_shifted / sigma_neg)
		area_neg = percent_below * (cdf_0_neg - cdf_min_neg)
		cdf_max_pos = norm.cdf(interval_max_shifted / sigma_pos)
		cdf_0_pos = norm.cdf(0 / sigma_pos)
		area_pos = percent_above * (cdf_max_pos - cdf_0_pos)
		area = area_neg + area_pos
	# Determine plotting window around the interval/sigmas so narrow ranges (e.g. -1..1) stay visible
	candidates = []
	if not np.isinf(interval_min_shifted):
		candidates.append(interval_min_shifted)
	if not np.isinf(interval_max_shifted):
		candidates.append(interval_max_shifted)
	# Add a few sigma multiples to see the tails
	candidates.extend([-4 * sigma_neg, 4 * sigma_pos, -4 * sigma_pos, 4 * sigma_neg])
	# Fallback if everything is inf/empty
	if len(candidates) == 0:
		candidates = [-1.0, 1.0]

	base_min = min(candidates)
	base_max = max(candidates)
	if np.isclose(base_min, base_max):
		base_min -= 1.0
		base_max += 1.0

	padding = max(0.1, 0.1 * (base_max - base_min))
	x_min = base_min - padding
	x_max = base_max + padding

	# Plot
	x_vals = np.linspace(x_min, x_max, 600)
	y_vals = split_pdf(x_vals)
	plt.figure(figsize=(8, 4))
	plt.plot(x_vals, y_vals)
	plt.fill_between(x_vals, 0, y_vals, alpha=0.3, label=f'PDF (p+={percent_above:.2f}, p-={percent_below:.2f})')
	# Handle infinite intervals for plotting
	raw_plot_min = interval_min_shifted if not np.isinf(interval_min_shifted) else x_min
	raw_plot_max = interval_max_shifted if not np.isinf(interval_max_shifted) else x_max
	plot_min = max(x_min, raw_plot_min)
	plot_max = min(x_max, raw_plot_max)
	plt.axvspan(plot_min, plot_max, color='black', alpha=0.4, label='Interval')
	# Create title with proper inf handling
	interval_str = f"[{interval_min_shifted:.4f}, {interval_max_shifted:.4f}]"
	if np.isinf(interval_min_shifted):
		interval_str = f"[-∞, {interval_max_shifted:.4f}]"
	if np.isinf(interval_max_shifted):
		interval_str = f"[{interval_min_shifted:.4f}, ∞]"
	if np.isinf(interval_min_shifted) and np.isinf(interval_max_shifted):
		interval_str = "[-∞, ∞]"
	plt.title(f'Feature: {feature_key} | Cost: {area:.4f} | Interval: {interval_str} | σ+={sigma_pos:.2f}, σ-={sigma_neg:.2f}')
	plt.axvline(0, color='black', linestyle='--')
	plt.xlabel('Shifted value (relative to sample)')
	plt.ylabel('Probability density')
	plt.legend()
	plt.tight_layout()
	output_path = Path(output_dir) / f'feature_{feature_key}_cost_plot.png'
	plt.savefig(output_path)
	plt.close()
	print(f"  Saved plot to {output_path}")

def plot_extreme_costs_per_category(cost_df, db, tests_sample, sigmas_all, top_n=2, output_dir='fig'):
	"""
	Per ogni categoria (reason_type), plotta le prime N con costo minimo e massimo.
	Parameters
	----------
	cost_df : pd.DataFrame
	DataFrame con colonne: sample_id, bitmap_index, reason_type, cost, icf
	db : dict
	Database con le informazioni (reasons, non_reasons, anti_reasons)
	tests_sample : dict
	Dizionario con i sample test
	sigmas_all : dict
	Sigmas calcolati per ogni sample
	top_n : int
	Numero di reason da plottare per estremo (default: 2)
	output_dir : str
	Directory per salvare i plot (default: 'fig')
	"""
	from pathlib import Path
	# Create output directory
	Path(output_dir).mkdir(parents=True, exist_ok=True)
	# Raggruppa per reason_type
	reason_types = cost_df['reason_type'].unique()
	print(f"\n{'='*80}")
	print(f"PLOTTING EXTREME COSTS FOR EACH CATEGORY (Top {top_n} min and max)")
	print(f"{'='*80}\n")
	for reason_type in reason_types:
		print(f"\n{'─'*80}")
		print(f" Category: {reason_type.upper()}")
		print(f"{'─'*80}")
		# Filtra per categoria
		subset = cost_df[cost_df['reason_type'] == reason_type].copy()
		if len(subset) == 0:
			print(f"  No data for {reason_type}")
			continue
		# Ordina per costo
		subset_sorted = subset.sort_values('cost')
		# Prendi top N min e max
		n_available = len(subset_sorted)
		n_min = min(top_n, n_available)
		n_max = min(top_n, n_available)
		min_costs = subset_sorted.head(n_min)
		max_costs = subset_sorted.tail(n_max)
		print(f"\n TOP {n_min} MINIMUM COST:")
		for idx, (_, row) in enumerate(min_costs.iterrows(), 1):
			sample_id = row['sample_id']
			bitmap_idx = row['bitmap_index']
			cost = row['cost']
			icf = row['icf']
			print(f"\n  #{idx} - Sample: {sample_id}, Bitmap: {bitmap_idx}, Cost: {cost:.6f}")
			# Get sample and sigmas
			if sample_id not in tests_sample:
				print(f"    Warning: sample_id {sample_id} not found in tests_sample")
				continue
			if sample_id not in sigmas_all:
				print(f"    Warning: sample_id {sample_id} not found in sigmas_all")
				continue
			sample = tests_sample[sample_id]['features']
			sigmas = sigmas_all[sample_id]

			# Plot tutte le feature dell'ICF
			if icf and len(icf) > 0:
				feature_dir = f"{output_dir}/{reason_type}_min_{idx}"
				Path(feature_dir).mkdir(parents=True, exist_ok=True)

				print(f"       Plotting {len(icf)} features to {feature_dir}/")
				for feature_key in icf.keys():
					try:
						plot_cost_distribution(
							sample, icf, sigmas, feature_key,
							output_dir=feature_dir
						)
					except Exception as e:
						print(f"        Warning: could not plot {feature_key}: {e}")
		print(f"\n TOP {n_max} MAXIMUM COST:")
		for idx, (_, row) in enumerate(max_costs.iterrows(), 1):
			sample_id = row['sample_id']
			bitmap_idx = row['bitmap_index']
			cost = row['cost']
			icf = row['icf']
			print(f"\n  #{idx} - Sample: {sample_id}, Bitmap: {bitmap_idx}, Cost: {cost:.6f}")
			# Get sample and sigmas
			if sample_id not in tests_sample:
				print(f"    Warning: sample_id {sample_id} not found in tests_sample")
				continue
			if sample_id not in sigmas_all:
				print(f"    Warning: sample_id {sample_id} not found in sigmas_all")
				continue
			sample = tests_sample[sample_id]['features']
			sigmas = sigmas_all[sample_id]

			# Plot tutte le feature dell'ICF
			if icf and len(icf) > 0:
				feature_dir = f"{output_dir}/{reason_type}_max_{idx}"
				Path(feature_dir).mkdir(parents=True, exist_ok=True)

				print(f"       Plotting {len(icf)} features to {feature_dir}/")
				for feature_key in icf.keys():
					try:
						plot_cost_distribution(
							sample, icf, sigmas, feature_key,
							output_dir=feature_dir
						)
					except Exception as e:
						print(f"        Warning: could not plot {feature_key}: {e}")
	print(f"\n{'='*80}")
	print(f" Completed plotting for all categories")
	print(f"{'='*80}\n")

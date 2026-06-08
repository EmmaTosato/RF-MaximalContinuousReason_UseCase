#!/usr/bin/env python3
"""
Initialize Redis with Baseline Pre-trained Classifiers

This script:
1. Connects to Redis (clearing DBs by default)
2. Loads a pre-trained Random Forest from baseline/Classifiers-100-converted
3. Loads dataset from baseline/resources/datasets
4. Loads test samples from baseline/resources/datasets
5. Stores everything in Redis (Dataset, Forest, Endpoints, Initial Candidate)

Directory Structure:
- Classifiers: baseline/Classifiers-100-converted/<dataset_name>/*.json
- Datasets: baseline/resources/datasets/<dataset_name>/<dataset_name>.csv
- Samples: baseline/resources/datasets/<dataset_name>/<dataset_name>.samples

Usage:
    python init_baseline.py --list-datasets
    python init_baseline.py iris --class-label "0"
    python init_baseline.py sonar --class-label "1" --test-sample-index "0,5-8,20"
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import redis
import json
import datetime
from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Shared modules
from redis_helpers.connection import connect_redis
from redis_helpers.utils import clean_all_databases

from init_utils import (
    store_training_set,
    store_forest_and_endpoints,
    process_all_classified_samples,
    initialize_seed_candidate
)
from helpers import convert_numpy_types, parse_sample_indices
from load_rf_from_json import load_rf_from_json
from baseline.xrf import Dataset
from rf_utils import sklearn_forest_to_forest
from rcheck_cache import rcheck_cache, saturate
from icf_eu_encoding import bitmap_mask_to_string, icf_to_bitmap_mask
from redis_helpers.samples import store_sample


# Constants
CLASSIFIERS_ROOT = os.path.join('baseline', 'Classifiers-100-converted')
DATASETS_ROOT = os.path.join('baseline', 'resources', 'datasets')
DATASETS = ['spraydryer','ann-thyroid', 'appendicitis', 'banknote', 'biodegradation', 'ecoli', 'glass2', 'heart-c', 'ionosphere', 'iris', 'karhunen', 'letter', 'magic', 'mofn-3-7-10', 'new-thyroid', 'pendigits', 'phoneme', 'ring', 'segmentation', 'shuttle', 'sonar', 'spambase', 'spectf', 'texture', 'threeOf9', 'twonorm', 'vowel', 'waveform-21', 'waveform-40', 'wdbc', 'wine-recog', 'wpbc', 'xd6']



def load_dataset_from_baseline(dataset_name, separator=',', train_idx=None):
    """
    Load dataset CSV and samples from baseline directory structure.

    Args:
        dataset_name: Name of the dataset
        separator: CSV separator (default: ',')
        train_idx: Optional list of row indices into the CSV to use as training set.
            If provided, builds X_train from those rows instead of a random split.
            Use this when the fold split is known (e.g. from _meta.json).

    Returns:
        (X_train, X_test, y_train, y_test, feature_names, all_classes)
    """
    # Handle potential name mismatch (underscore vs hyphen)
    actual_name = dataset_name
    dataset_dir = os.path.join(DATASETS_ROOT, dataset_name)

    if not os.path.exists(dataset_dir) and '_' in dataset_name:
        alt_name = dataset_name.replace('_', '-')
        if os.path.exists(os.path.join(DATASETS_ROOT, alt_name)):
            actual_name = alt_name

    dataset_path = os.path.join(DATASETS_ROOT, actual_name, f"{actual_name}.csv")
    samples_path = os.path.join(DATASETS_ROOT, actual_name, f"{actual_name}.samples")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset CSV not found: {dataset_path}")

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Samples file not found: {samples_path}")

    if os.path.getsize(samples_path) == 0:
        raise ValueError(f"Samples file is empty: {samples_path}")

    # Load dataset using baseline Dataset class
    print(f"[INFO] Loading dataset from: {dataset_path}")
    data = Dataset(filename=dataset_path, separator=separator, use_categorical=False)

    if train_idx is not None:
        # Use the exact fold split from _meta.json — avoids sigma contamination
        csv_df = pd.read_csv(dataset_path, sep=separator)
        feature_cols = data.features
        X_train_raw = csv_df.iloc[train_idx][feature_cols].values
        y_train = csv_df.iloc[train_idx]['label'].values
        X_train = data.transform(X_train_raw)
        print(f"[INFO] Using fold train_idx: {len(train_idx)} training samples")
    else:
        # Fallback: random 80/20 split (UCI/classic baseline datasets)
        X_train_raw, _, y_train, _ = data.train_test_split()
        X_train = data.transform(X_train_raw)

    # Load samples - these will be our actual test samples
    print(f"[INFO] Loading samples from: {samples_path}")
    samples = np.loadtxt(samples_path, delimiter=separator)
    samples = np.atleast_2d(samples)

    # Validate sample dimensions
    expected_features = len(data.features)
    if samples.shape[1] == expected_features + 1:
        print("[INFO] Sample file includes labels; dropping last column.")
        samples = samples[:, :-1]
    elif samples.shape[1] != expected_features:
        raise ValueError(
            f"Sample file feature count mismatch: expected {expected_features}, "
            f"found {samples.shape[1]} in {samples_path}"
        )

    print(f"[INFO] Loaded {len(samples)} test samples")
    print(f"[INFO] Training set: {len(X_train)} samples")
    print(f"[INFO] Features: {data.features}")
    print(f"[INFO] Classes: {np.unique(y_train)}")

    return X_train, samples, y_train, None, data.features, np.unique(y_train), data


def load_classifier_from_pkl(model_file):

    print(f"[INFO] Loading pre-trained classifier: {os.path.basename(model_file)}")

    with open(model_file, "rb") as f:
        sklearn_rf = pickle.load(f)

    print(f"[INFO] Successfully loaded classifier with {sklearn_rf.n_estimators} trees")

    return sklearn_rf


def process_all_classified_samples_baseline(
    connections,
    dataset_name,
    class_label,
    our_forest,
    X_test,
    eu_data
):
    """
    Process all test samples that are classified with the specified class label.
    Store samples in DATA and their ICF representations in R.

    Args:
        connections: Redis connections dict
        dataset_name: Name of the dataset
        class_label: Target class label to filter
        our_forest: Custom Forest object
        X_test: Test features array
        eu_data: Endpoints universe data
    Returns:
        tuple: (stored_samples list, summary dict)
    """
    print(f"\n=== Processing All Samples Classified as '{class_label}' ===")

    # Find all test samples that are classified as the target class
    target_samples_data = []
    current_time = datetime.datetime.now().isoformat()

    # Apply sample percentage filtering if specified


    for i, sample in enumerate(X_test):
        predicted_label = our_forest.predict(sample)

        # Store ALL samples classified with the target label (regardless of correctness)
        if predicted_label == class_label:
            target_samples_data.append({
                'test_index': i,
                'sample_dict': sample,
                'predicted_label': predicted_label,
            })

    print(f"Found {len(target_samples_data)} samples classified as '{class_label}'")

    if len(target_samples_data) == 0:
        print("[WARNING] No samples classified with the target label!")
        return [], {}

    # Store all samples and their ICF representations
    stored_samples = []
    correct_predictions = 0

    for idx, sample_data in enumerate(target_samples_data):
        sample_key = f"sample_{dataset_name}_{class_label}_{idx}"

        # Store sample in DATA with full metadata
        data_entry = {
            'sample_dict': sample_data['sample_dict'],
            'predicted_label': sample_data['predicted_label'],
            'test_index': sample_data['test_index'],
            'dataset_name': dataset_name,
            'timestamp': current_time,
        }

        # Store sample using our helper function
        if store_sample(connections['DATA'], sample_key, sample_data['sample_dict']):
            # Also store full metadata separately
            connections['DATA'].set(f"{sample_key}_meta", json.dumps(data_entry))

        # Calculate ICF and store in R
        try:
            sample_icf = our_forest.extract_icf(sample_data['sample_dict'])
            icf_bitmap = bitmap_mask_to_string(icf_to_bitmap_mask(sample_icf, eu_data))

            # Store ICF bitmap in R with metadata
            icf_metadata = {
                'sample_key': sample_key,
                'dataset_name': dataset_name,
                'class_label': class_label,
                'test_index': sample_data['test_index'],
                'timestamp': current_time
            }

            connections['R'].set(icf_bitmap, json.dumps(icf_metadata))

            stored_samples.append({
                'sample_key': sample_key,
                'icf_bitmap': icf_bitmap,
                'test_index': sample_data['test_index']
            })

        except Exception as e:
            print(f"[WARNING] Failed to process sample {idx}: {e}")
            continue

    # Store summary information
    summary = {
        'dataset_name': dataset_name,
        'target_class_label': class_label,
        'total_samples_processed': len(stored_samples),
        'total_test_samples': len(X_test),
        'samples_with_target_label': len(target_samples_data),
        'timestamp': current_time,
        'sample_keys': [s['sample_key'] for s in stored_samples]
    }

    connections['DATA'].set(f"summary_{dataset_name}_{class_label}", json.dumps(summary))

    print(f"[OK] Stored {len(stored_samples)} samples in DATA")
    print(f"[OK] Stored {len(stored_samples)} ICF representations in R")
    print(f"[OK] Summary stored in DATA['summary_{dataset_name}_{class_label}']")

    return stored_samples, summary


def main():
    parser = argparse.ArgumentParser(
        description='Initialize Redis with DSD Pre-trained Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python init_baseline.py --list-datasets

  # Initialize with iris dataset
  python init_baseline.py iris --class-label "0"

  # Use specific test samples
  python init_baseline.py sonar --class-label "1" --test-sample-index "0,5,10"

  # Use sample ranges
  python init_baseline.py iris --class-label "0" --test-sample-index "0-10,20-30"

  # Select a different classifier if multiple exist
  python init_baseline.py iris --class-label "0" --classifier-index 1
        """
    )
    parser.add_argument('dataset_name', nargs='?', 
                       help='Name of the baseline dataset (e.g., iris, sonar)')
    parser.add_argument('--model-file', type=str, required=True, help='model pickle file')
    
    # Core arguments
    parser.add_argument('--redis-port', type=int, default=6379,
                       help='Redis port (default: 6379)')
    parser.add_argument('--class-label', type=str, required=False,
                       help='Target class label to process (required if dataset is provided)')
    parser.add_argument('--test-sample-index', type=str, default=None,
                       help='Index or indices of samples to use (e.g., "0,5,10" or "0-10")')
    parser.add_argument('--no-clean', action='store_true',
                       help='Do not clean Redis databases before initialization')
    #parser.add_argument('--preserve-ar', action='store_true',
    #                   help='Preserve AR database (DB5) when cleaning (implies --no-clean for DB5 only)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
       
    if not args.class_label:
        print("[ERROR] --class-label is required")
        return 1
    
    print(f"\n[START] Initializing Random Path Worker System (Baseline)")
    print(f"[INFO] Dataset: {args.dataset_name}")
    print(f"[INFO] Target Class Label: {args.class_label}")
    
    try:
        # 1. Connect to Redis
        connections, db_mapping = connect_redis(port=args.redis_port)
        if not connections:
            return 1
        
        clean_all_databases(connections, db_mapping)
        
        # 2. Load meta file if present (provides fold splits, subject_id, actual_label)
        meta_path = Path(DATASETS_ROOT) / args.dataset_name / f"{args.dataset_name}_meta.json"
        fold_meta = None
        if meta_path.exists():
            with open(meta_path) as f:
                fold_meta = json.load(f)
            print(f"[INFO] Loaded fold meta: {meta_path}")
        else:
            print(f"[WARNING] No meta file found at {meta_path}; falling back to random split")

        train_idx = fold_meta["train_idx"] if fold_meta is not None else None

        # 3. Load Dataset
        X_train, X_test_samples, y_train, _, feature_names, all_classes, data = load_dataset_from_baseline(
            args.dataset_name, train_idx=train_idx
        )

        # Build y_test and subject_ids from meta (parallel to X_test_samples rows)
        if fold_meta is not None:
            samples_meta = fold_meta["samples"]
            y_test = np.array([samples_meta[str(i)]["actual_label"] for i in range(len(X_test_samples))])
            subject_ids = [samples_meta[str(i)]["subject_id"] for i in range(len(X_test_samples))]
            print(f"[INFO] y_test and subject_ids loaded from meta: {len(subject_ids)} subjects")
        else:
            y_test = None
            subject_ids = None
                
        print(f"[INFO] Dataset: {args.dataset_name}")
        print(f"[INFO] Features: {len(feature_names)}")
        print(f"[INFO] Training samples: {len(X_train)}")
        print(f"[INFO] Test samples: {len(X_test_samples)}")
        print(f"[INFO] Classes: {all_classes}")
        
        # 3. Load Pre-trained Classifier
        #sklearn_rf e` un oggetto RandomForestClassifier di sklearn
        #sklearn_rf, classifier_path = load_classifier_from_json(args.dataset_name)
        sklearn_rf = load_classifier_from_pkl(args.model_file)
             
        our_forest = sklearn_forest_to_forest(sklearn_rf, feature_names)
        
        # Normalize class_label to match forest prediction format (e.g. '1' → '1.0')
        forest_labels = [str(c) for c in sklearn_rf.classes_]
        if args.class_label not in forest_labels:
            for fl in forest_labels:
                try:
                    if float(fl) == float(args.class_label):
                        args.class_label = fl
                        print(f"[INFO] Normalizing class label to '{fl}'")
                        break
                except (ValueError, TypeError):
                    pass
        
        # Store training set in DATA database
        training_set_stored = store_training_set(connections, X_train, y_train, feature_names, args.dataset_name, dataset_type='dsd')
        print("\n[INFO] Training set stored: ", training_set_stored)
        
        
        # 4. Store Training Set Metadata        
        print("\n[INFO] Storing forest and computing endpoints...")
        eu_data = store_forest_and_endpoints(connections, our_forest)
        

        # 5. Process Test Samples
        print("\n[INFO] Processing test samples...")

        if fold_meta is None:
            raise FileNotFoundError(
                f"Meta file not found: {meta_path}. "
                f"Run export_to_drifts.py to generate it."
            )

        stored_samples, summary = process_all_classified_samples(
            connections,
            args.dataset_name,
            args.class_label,
            our_forest,
            X_test_samples,
            y_test,
            feature_names,
            eu_data,
            subject_ids=subject_ids,
            dataset_type='dsd',
        )
        

        if not stored_samples:
            print("[WARNING] No samples processed, cannot initialize seed candidate.")
            return 1


        print("\n[INFO] Initializing seed candidates...")
        for s in stored_samples:
            meta_json = connections['DATA'].get(f"{s['sample_key']}_meta")
            if meta_json:
                meta = json.loads(meta_json)
                initialize_seed_candidate(connections, meta, our_forest, eu_data)
            else:
                print("[WARNING] No meta data found for a sample")
                
        # 9. Store target label for worker compatibility
        connections['DATA'].set('label', args.class_label)
        print(f"[INFO] Target label '{args.class_label}' set for worker processing")
        
        # 10. Store classifier metadata
        metadata = {
            'dataset': args.dataset_name,
            'classifier_path': args.model_file,
            'n_estimators': sklearn_rf.n_estimators,
            'max_depth': sklearn_rf.max_depth,
            'n_features': sklearn_rf.n_features_in_,
            'classes': list(sklearn_rf.classes_.astype(str)),
            'timestamp': datetime.datetime.now().isoformat()
        }
        connections['DATA'].set('classifier_metadata', json.dumps(metadata))
        
        print(f"\n[SUCCESS] Successfully initialized {args.dataset_name}")
        print(f"[SUCCESS] Pre-trained classifier loaded from: {os.path.basename(args.model_file)}")
        print(f"[SUCCESS] Ready for worker processing with {len(stored_samples)} samples")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n[ABORT] Initialization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

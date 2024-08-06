import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
import joblib
import json
from glob import glob
import argparse

def load_and_subsample(path: str) -> np.ndarray:
    """
    Load and subsample a waveform file.

    Args:
        path (str): Path to the waveform file.

    Returns:
        numpy.ndarray: Subsampled waveform data.
    """
    try:
        # Load with np.float64 for high precision in intermediate steps, but consider final dtype based on your needs
        arr = np.expand_dims(np.load(path)[1::2, :].astype(np.float64), axis=0)
        return arr
    except Exception as e:
        print(f"Failed to load {path} with error: {e}")
        return np.array([])  # Return an empty array in case of failure

def parallel_load_files(paths: list) -> np.ndarray:
    """
    Load and subsample waveform files in parallel.

    Args:
        paths (list): List of paths to the waveform files.

    Returns:
        numpy.ndarray: Concatenated subsampled waveform data.
    """
    results = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(load_and_subsample, path): path for path in paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Loading and subsampling waveforms"):
            result = future.result()
            if result.size > 0:  # Ensure non-empty arrays are appended
                results.append(result)
    return np.vstack(results)

def filter_numericals(arr: np.ndarray) -> np.ndarray:
    """
    Filter out non-numerical arrays.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        numpy.ndarray: Filtered array containing only numerical arrays.
    """
    return np.array([a for a in tqdm(arr, desc="Filtering non-numerical arrays") if np.isfinite(a).all()])

def compute_scaling_params(arr: np.ndarray) -> StandardScaler:
    """
    Compute scaling parameters (mean and standard deviation) for the input array.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        sklearn.preprocessing.StandardScaler: Scaler object with computed parameters.
    """
    scaler = StandardScaler()
    scaler.mean_ = np.array([np.nanmean(arr)])
    scaler.scale_ = np.array([np.nanstd(arr)])
    scaler.var_ = np.array([scaler.scale_ ** 2])
    scaler.n_features_in_ = 1
    return scaler

def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process waveform data")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing waveform data")
    return parser.parse_args()

def main():
    """
    Main function to process waveform data and compute scaling parameters.
    """
    args = parse_arguments()
    data_dir = args.data_dir
    
    waveform_paths = glob(f"{data_dir}/**/*.npy", recursive=True)
    all_waveforms = parallel_load_files(waveform_paths)
    print("Shape of the concatenated array:", all_waveforms.shape)
    
    all_waveforms_filtered = filter_numericals(all_waveforms)
    flattened_waveforms = all_waveforms_filtered.reshape(-1, 1)

    scaler = compute_scaling_params(flattened_waveforms)
    joblib.dump(scaler, 'scaler.pkl')

    print(f"Mean: {scaler.mean_[0]}\nStddev: {scaler.scale_[0]}")

if __name__ == "__main__":
    main()
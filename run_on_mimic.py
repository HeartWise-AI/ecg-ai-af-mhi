import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('HF_TOKEN')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/results')  # Default to /results if not specified

class ECGDataset:
    def __init__(self, file_paths, mean, std, batch_size=32):
        self.file_paths = file_paths
        self.mean = mean
        self.std = std
        self.batch_size = batch_size

    def load_and_preprocess(self, file_path):
        try:
            ecg_array = np.load(file_path.numpy().decode())
            if not np.isfinite(ecg_array).all():
                return np.zeros((2500, 12), dtype=np.float32), file_path  # Return zero array if not numerical
            
            # Downsample mimic data
            ecg_array = ecg_array[::2, :]
            
            # Switch channel aVL & aVF (model trained on ICM data)
            ecg_array[:, [4, 5]] = ecg_array[:, [5, 4]]
            
            ecg_array = (ecg_array - self.mean) / self.std
            return ecg_array.astype(np.float32), file_path
        except:
            return np.zeros((2500, 12), dtype=np.float32), file_path  # Return zero array if loading fails

    def create_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(self.file_paths)
        dataset = dataset.map(
            lambda x: tf.py_function(self.load_and_preprocess, [x], [tf.float32, tf.string]),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

def run_on_MIMIC(params, model):
    # Check available GPU
    print("Available GPUs:", tf.config.list_physical_devices('GPU'))

    # Set normalization parameters
    mean = params['model_config']['MIMIC']['mean']
    std = params['model_config']['MIMIC']['std']
    
    # Load and preprocess the DataFrame
    df = pd.read_csv("pred_waveform_path.csv.gz")
    print(len(df))
    df['waveform_path'] = df['waveform_path'].str.replace("/media/data1/ravram/MIMIC-IV/1.0/files/", "/mimic-IV/")
    
    # Create the dataset
    batch_size = 32
    ecg_dataset = ECGDataset(df['waveform_path'].tolist(), mean, std, batch_size=batch_size)
    tf_dataset = ecg_dataset.create_dataset()
    
    # Process the data and make predictions
    predictions = []
    paths = []
    for batch_data, batch_paths in tqdm(tf_dataset, total=len(df)//batch_size + 1, desc="Processing batches"):
        batch_predictions = model(batch_data)
        predictions.extend(batch_predictions.numpy())
        paths.extend([p.numpy().decode() for p in batch_paths])

    # Ensure paths and predictions have the same length
    assert len(paths) == len(predictions), f"Mismatch in number of paths ({len(paths)}) and predictions ({len(predictions)})"
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'waveform_path': paths,
        'pred': predictions
    })
    
    return results_df
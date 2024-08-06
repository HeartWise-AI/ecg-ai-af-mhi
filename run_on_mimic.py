import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from dotenv import load_dotenv

from data.mimic_dataset import MimicDataset

load_dotenv()
API_KEY = os.getenv('HF_TOKEN')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/results')  # Default to /results if not specified

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
    ecg_dataset = MimicDataset(df['waveform_path'].tolist(), mean, std, batch_size=batch_size)
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
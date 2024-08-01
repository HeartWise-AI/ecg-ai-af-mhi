
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from tensorflow.keras.models import load_model

load_dotenv()
API_KEY = os.getenv('HF_TOKEN')

if __name__ == "__main__":
       
    snapshot_download(
        repo_id="heartwise/ecgAI_AF_MHI", 
        local_dir="weights"
    )
    
    # Importing the model
    model = load_model(
            f"weights/best_model.h5", 
            custom_objects={'Addons>F1Score': tfa.metrics.F1Score}
        )
        
    # import normalization config
    with open("weights/config.json", "r") as f:
        config = json.load(f)
    
    mean = config["mean"]
    std = config["std"]
    
    # Read mimic dataset    
    df = pd.read_csv("mimic_index.with_bandwidth.with_filtering.csv")

    batch_size = 32
    predictions = []
    paths = []

    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_ecgs = []
        for j in range(i, min(i + batch_size, len(df)), 1):
            # Get ecg array path
            np_path = df['waveform_path'].iloc[j]
            paths.append(np_path)
            
            # Load ecg_array
            ecg_array = np.load(np_path)
            
            # Downsample mimic data
            step = 2 # 5000 (MIMIC) // 2500 (ICM)
            ecg_array = ecg_array[::step, :]
            
            # Switch channel aVL & aVF (model trained on ICM data)
            ecg_array[:, 4], ecg_array[:, 5] = ecg_array[:, 5], ecg_array[:, 4]
            
            ecg_array = (ecg_array - mean) / std
            
            batch_ecgs.append(ecg_array.astype(np.float16))
            
        # convert list to tf tensor batch
        tf_tensor_batch = tf.convert_to_tensor(batch_ecgs)
        
        # Run Inference
        y_pred = model(tf_tensor_batch)
        
        predictions.extend(y_pred.numpy())
        
    # Save predictions to DataFrame
    results_df = pd.DataFrame({
        'waveform_path': paths,
        'pred': predictions
    })
    
    # Save DataFrame to CSV
    results_df.to_csv("predictions.csv", index=False)     

    












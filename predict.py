import xmltodict
import base64
import struct
import json
import os

import numpy as np 
import pandas as pd 
from glob import glob
import argparse 
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from huggingface_hub import snapshot_download
from utils import extract_wf_as_npy
            
def make_inference(config):
    
    # Importing the model
    model = load_model(
            config['model_path'], 
            custom_objects={'Addons>F1Score': tfa.metrics.F1Score}
        )

    # Load the data
    xml_files, waveforms = extract_wf_as_npy(config['xml_dir'])
    
    # Standard scaler
    mean = config['mean']
    std = config['std']
    waveforms_std = (waveforms - mean) / std
    waveforms_std = waveforms_std.astype(np.float16)

    # Model predictions
    y_pred = model(waveforms_std)
    
    df = pd.DataFrame({
            "xml_file" : xml_files,
            "logit" : y_pred.numpy().flatten()
        })
    
    return(df)

    
def get_arguments():
    """
    Generate the arguments using CLI
    """
    parser = argparse.ArgumentParser(
            description="Get argument", 
            add_help=False
        )
    
    parser.add_argument(
        "--config",
        metavar="config",
        type=str,
        help="Enter path to xml",
    )
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_arguments()
    try:
        snapshot_download(
            repo_id="heartwise/ecgAI_AF_MHI", 
            local_dir="./weights/"
        )
    except Exception as e:
        print("Error: Could not download the model weights. \nPlease make sure you are authenticated and have access to the resource.")
        exit(1)  # Abort the script with a non-zero exit code to indicate an error
    
        
    with open(args.config) as f:
        config = json.load(f)
        
    results = make_inference(config)
    print(results)
    
    
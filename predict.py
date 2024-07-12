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
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('HF_TOKEN')

def make_inference(config):
    
    # Importing the model
    model = load_model(
            f"{config['model_path']}/best_model.h5", 
            custom_objects={'Addons>F1Score': tfa.metrics.F1Score}
        )

    # Load the data
    xml_files, waveforms = extract_wf_as_npy(config['xml_dir'])
    
    # Standard scaler
    mean = config['model_config']['mean']
    std = config['model_config']['std']
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
    with open(args.config) as f:
        params = json.load(f)
    
    
    ## Download the weights only if not already present in the weights folder
    if not os.path.exists(os.path.join(params['model_path'], 'best_model.h5')):
        try:
            snapshot_download(
                repo_id="heartwise/ecgAI_AF_MHI", 
                local_dir=params['model_path']
            )
        except Exception as e:
            print(e)
            exit(1)  # Abort the script with a non-zero exit code to indicate an error
    with open(f"{params['model_path']}/config.json") as f:
        config = json.load(f)
    
    params['model_config'] = config
    
    results = make_inference(params)
    results.to_csv(f"{params['output_dir']}/results.csv",index=False)
    
    
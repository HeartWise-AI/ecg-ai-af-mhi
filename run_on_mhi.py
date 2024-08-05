
import numpy as np
import pandas as pd
from utils import extract_wf_as_npy

def run_on_MHI(config, model):
    
    # Load the data
    xml_files, waveforms = extract_wf_as_npy(config['xml_dir'])
    
    # Standard scaler
    waveforms_std = (waveforms - config['model_config']['MHI']['mean']) / config['model_config']['MHI']['std']
    waveforms_std = waveforms_std.astype(np.float16)

    # Model predictions
    y_pred = model(waveforms_std)
    
    df = pd.DataFrame({
            "xml_file" : xml_files,
            "logit" : y_pred.numpy().flatten()
        })
    
    return df
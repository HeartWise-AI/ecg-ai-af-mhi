import os
import json
import argparse
import tensorflow_addons as tfa

from dotenv import load_dotenv
from run_on_mhi import run_on_MHI
from run_on_mimic import run_on_MIMIC
from huggingface_hub import snapshot_download
from tensorflow.keras.models import load_model


load_dotenv()
api_key = os.getenv('HF_TOKEN')
    
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
        help="Enter path to config file",
    )
    parser.add_argument(
        "--MIMIC",
        action='store_true',
        help="Run on MIMIC dataset",
    )
    parser.add_argument(
        "--MHI",
        action='store_true',
        help="Run on MHI dataset",
    )
    return parser.parse_args()

if __name__ == '__main__':
    
    
    args = get_arguments()
    with open(args.config) as f:
        params = json.load(f)
    
    # Download the weights only if not already present in the weights folder
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

    # Importing the model
    model = load_model(
            f"{config['model_path']}/best_model.h5", 
            custom_objects={'Addons>F1Score': tfa.metrics.F1Score}
        )
    
    params['model_config'] = config

    results = None   
    if args.MIMIC:
        results = run_on_MIMIC(params, model)
    elif args.MHI:
        results = run_on_MHI(params, model)
    else:
        raise ValueError("Unknow dataset, process on MIMIC or MHI")

    results.to_csv(f"{params['output_dir']}/results.csv",index=False)

    print(f"Processing complete. Results saved to {params['output_dir']}/results.csv")
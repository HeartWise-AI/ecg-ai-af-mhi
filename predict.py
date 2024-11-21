import os
import json
import argparse
import tensorflow_addons as tfa
import pandas as pd
from glob import glob
from ECGDataset import ECGDatasetMIMIC, ECGDatasetMHI
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from tensorflow.keras.models import load_model


load_dotenv()
api_key = os.getenv("HF_TOKEN")


def get_arguments():
    """
    Generate the arguments using CLI.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Get argument", add_help=False)

    parser.add_argument(
        "--config",
        metavar="config",
        type=str,
        help="Enter path to config file",
    )
    parser.add_argument(
        "--MIMIC",
        action="store_true",
        help="Run on MIMIC dataset",
    )
    parser.add_argument(
        "--MHI",
        action="store_true",
        help="Run on MHI dataset",
    )
    parser.add_argument(
        "--csv_data",
        action="store_true",
        help="Pass the ECGs through a txt file of paths"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    with open(args.config) as f:
        params = json.load(f)

    # Download the weights only if not already present in the weights folder
    if not os.path.exists(os.path.join(params["model_path"], "best_model.h5")):
        try:
            snapshot_download(
                repo_id="heartwise/ecgAI_AF_MHI", local_dir=params["model_path"]
            )
        except Exception as e:
            print(e)
            exit(1)  # Abort the script with a non-zero exit code to indicate an error

    with open(f"{params['model_path']}/config.json") as f:
        config = json.load(f)

    # Importing the model
    model = load_model(
        f"{params['model_path']}/best_model.h5",
        custom_objects={"Addons>F1Score": tfa.metrics.F1Score},
    )

    params["model_config"] = config

    results = None
    
    if args.csv_data:
        waveform_files_df = pd.read_csv(params['csv_file'],header=None)
        waveform_files = [p for p in waveform_files_df[0]]
    
    if args.MIMIC:
        
        # Load and preprocess the DataFrame for MIMIC dataset
        if not args.csv_data:
            waveform_files = glob(f"{params['data_dir']['NPY']}/**/*.npy", recursive=True)
        # Create the MIMIC dataset
        ecg_dataset = ECGDatasetMIMIC(
            waveform_files,
            params["model_config"]["MIMIC"]["mean"],
            params["model_config"]["MIMIC"]["std"],
            batch_size=params["batch_size"],
        )
        result_filename = "results_mimic.csv"

    elif args.MHI:
        # Load and preprocess the DataFrame for MHI dataset
        if not args.csv_data:
            waveform_files = glob(f"{params['data_dir']['XML']}/**/*.xml", recursive=True)
        # Create the MHI dataset
        ecg_dataset = ECGDatasetMHI(
            waveform_files,
            params["model_config"]["MHI"]["mean"],
            params["model_config"]["MHI"]["std"],
            batch_size=params["batch_size"],
        )
        result_filename = "results_mhi.csv"
    else:
        raise ValueError("Unknown dataset, process on MIMIC or MHI")

    # Create the TensorFlow dataset
    tf_dataset = ecg_dataset.create_dataset()

    # Process the data and make predictions
    predictions = []
    paths = []
    for batch_data, batch_paths in tqdm(
        tf_dataset,
        total=len(waveform_files) // params["batch_size"] + 1,
        desc="Processing batches",
    ):
        batch_predictions = model(batch_data)
        predictions.extend(batch_predictions.numpy())
        paths.extend([p.numpy().decode() for p in batch_paths])

    # Ensure paths and predictions have the same length
    assert (
        len(paths) == len(predictions)
    ), f"Mismatch in number of paths ({len(paths)}) and predictions ({len(predictions)})"

    # Create results DataFrame
    results_df = pd.DataFrame({"waveform_path": paths, "pred": predictions})

    # Save the results DataFrame to a CSV file
    results_df.to_csv(f"{params['output_dir']}/{result_filename}", index=False)

    print(f"Processing complete. Results saved to {params['output_dir']}/{result_filename}")
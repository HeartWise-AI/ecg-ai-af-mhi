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
        "--csv_data",
        action="store_true",
        help="Pass the ECGs through a txt file of paths"
    )
    parser.add_argument(
        "--numpy",
        action="store_true",
        help="Pass the ECGs through a txt file of paths"
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Pass the ECGs through a txt file of paths"
    )
    return parser.parse_args()


if __name__ == "__main__":
    
    args = get_arguments()

    # Download the weights only if not already present in the weights folder
    if not os.path.exists("/weights/best_model.h5"):
        try:
            snapshot_download(
                repo_id="heartwise/ecgAI_AF_MHI", local_dir="/weights/"
            )
        except Exception as e:
            print(e)
            exit(1)  # Abort the script with a non-zero exit code to indicate an error

    with open(f"/weights/config.json") as f:
        config = json.load(f)

    # Importing the model
    model = load_model(
        f"/weights/best_model.h5",
        custom_objects={"Addons>F1Score": tfa.metrics.F1Score},
    )

    config = config
    results = None
    
    ecg_files = os.getenv("ECG_FILES")
    waveform_files_df = pd.read_csv(ecg_files,header=None)
    waveform_files = [p for p in waveform_files_df[0]]

    dataset = os.getenv("DATASET")
    batch_size = int(os.getenv("BATCH_SIZE"))
    if dataset=="MIMIC":
        # Create the MIMIC dataset
        ecg_dataset = ECGDatasetMIMIC(
            waveform_files,
            config["MIMIC"]["mean"],
            config["MIMIC"]["std"],
            batch_size=batch_size,
        )

    elif dataset=="MHI":
        # Create the MHI dataset
        ecg_dataset = ECGDatasetMHI(
            waveform_files,
            config["MHI"]["mean"],
            config["MHI"]["std"],
            batch_size=batch_size,
        )
    else:
        raise ValueError("Unknown dataset, process on MIMIC or MHI")

    # Create the TensorFlow dataset
    tf_dataset = ecg_dataset.create_dataset()

    # Process the data and make predictions
    predictions = []
    paths = []
    for batch_data, batch_paths in tqdm(
        tf_dataset,
        total=len(waveform_files) // batch_size + 1,
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
    results_df.to_csv(f"/results/results_{dataset}.csv", index=False)
    print(f"Processing complete. Results saved to results/results_{dataset}.csv")
import sys
import wfdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse

# Initialize argparse
parser = argparse.ArgumentParser(description="Process ECG data and extract features.")

# Define the flags for each argument the script accepts
parser.add_argument("--records", required=True, help="Path to the records CSV file")
parser.add_argument("--metadata", required=True, help="Path to the metadata CSV file")
parser.add_argument("--patients", required=True, help="Path to the patients CSV file")
parser.add_argument("--wf_path", required=True, help="Path to the waveform files")
parser.add_argument("--output_dir", required=True, help="Directory to save the outputs")
parser.add_argument("--output_file", required=True, help="Path to save the combined output CSV")

# Parse arguments
args = parser.parse_args()

def read_csv(path):
    """
    Read a CSV file, handling both compressed and uncompressed files.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded data as a DataFrame.
    """
    if path.endswith('gz'):
        return pd.read_csv(path, compression='gzip')
    else:
        return pd.read_csv(path)

# Load the data using the provided arguments
records = read_csv(args.records)
metadata = read_csv(args.metadata)
patients = read_csv(args.patients)

ecg_features = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']

def get_ecg(idx, output_file):
    """
    Process ECG data for a given index and save the results.

    Args:
        idx (int): Index of the ECG record.
        output_file (str): Path to the output file.
    """
    ecg_infos = records.iloc[idx]
    rec_path = f'{args.wf_path}{ecg_infos["path"]}'
    infos = rec_path.split('/')
    subject_id = infos[8][1:]
    study_id = infos[9][1:]
    
    ecg_time = pd.to_datetime(ecg_infos['ecg_time'])
    patient_info = patients[patients['subject_id']==int(subject_id)]
    if len(patient_info) > 0:
        age_at_ecg = ecg_time.year - patient_info.loc[patient_info.index[0], 'anchor_year'] + patient_info.loc[patient_info.index[0], 'anchor_age']
        gender = patient_info.loc[patient_info.index[0], 'gender']
    else:
        age_at_ecg = 'missing'
        gender = 'missing'
    
    rd_record = wfdb.rdrecord(rec_path)
    ecg_signal_uV = rd_record.p_signal * 1000
    
    meta = metadata[(metadata['subject_id'] == int(subject_id)) & (metadata['study_id'] == int(study_id))]
    reports = meta.filter(like='report_').values.ravel()
    concatenated_reports = ';'.join(str(x) for x in reports if pd.notna(x))
    output_str = f'{gender},{age_at_ecg},{concatenated_reports},{args.output_dir}{ecg_infos["path"]}.npy'
    
    for feature in ecg_features:
        if feature in meta.columns:
            output_str += f',{meta[feature].values[0]}'
        else:
            output_str += ','
    
    with open(output_file, 'a') as of:
        of.write(output_str + '\n')
        
    np.save(f'{args.output_dir}{ecg_infos["path"]}.npy', ecg_signal_uV)
    
if __name__ == '__main__':
    output_file = args.output_file
    for idx in tqdm(range(len(records))):
        get_ecg(idx, output_file)
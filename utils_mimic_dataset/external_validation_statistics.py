import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_auc_score

# Set up argument parser
parser = argparse.ArgumentParser(description="Process and analyze ECG data.")
# Define flags for each file path. Flags are made optional by using '--'
parser.add_argument("--ecg_csv", help="Path to the compressed ECG CSV file", required=True)
parser.add_argument("--patient_csv", help="Path to the compressed patient CSV file", required=True)
parser.add_argument("--pred_csv", help="Path to the CSV file with predictions", required=True)

# Parse arguments
args = parser.parse_args()

# load gz csv files based on arguments
df_ecg = pd.read_csv(args.ecg_csv)
df_patients = pd.read_csv(args.patient_csv)
df_pred = pd.read_csv(args.pred_csv)

print(f'Total Number of ECGs: {df_ecg.shape[0]}')
print(f'Total Number of Patients: {df_patients.shape[0]}')
print(f'Total Number of Predictions: {df_pred.shape[0]}')

# Correcting the typo mentioned in your original script
print(df_patients.shape)
print(df_pred.columns)

# Replace rows where is_within_5_years_of_af is null with 0
df_ecg['is_within_5_years_of_af'] = df_ecg['is_within_5_years_of_af'].fillna(0)
df_ecg['path'] = df_ecg['path'].str.replace('/media/data1/ravram/MIMIC-IV/1.0/files/','/mimic-IV/')

# Add pred column from df_pred to df_ecg
df_ecg = pd.merge(df_ecg, df_pred, how='inner', left_on='path', right_on='waveform_path')

# Convert pred column to float, safely remove square brackets
df_ecg['pred'] = df_ecg['pred'].str.replace('[', '').str.replace(']', '').astype(float)

# Filter df_ecg based on specified conditions
df_ecg = df_ecg[((df_ecg['sinus'] == 1) & ((df_ecg['days_from_af'] < 0) | (df_ecg['days_from_af'].isnull())) & (df_ecg['age_at_ecg'] >= 18) & (df_ecg['days_from_latest'] > 0))]

print(df_ecg.describe())

# Calculate AUC using is_within_5_years_of_af for label and pred for prediction
auc = roc_auc_score(df_ecg['is_within_5_years_of_af'], df_ecg['pred'])
print(f'AUC: {auc}')
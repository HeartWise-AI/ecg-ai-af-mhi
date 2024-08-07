import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.interpolate import UnivariateSpline
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.calibration import calibration_curve
from scipy.interpolate import UnivariateSpline
import seaborn as sns
from validation_utils import *
import os

def load_data(ecg_csv, patient_csv, pred_csv):
    if ecg_csv.endswith('gz'):
        df_ecg = pd.read_csv(ecg_csv, compression='gzip')
    else:
        df_ecg = pd.read_csv(ecg_csv)

    if patient_csv.endswith('gz'):
        df_patients = pd.read_csv(patient_csv, compression='gzip')
    else:
        df_patients = pd.read_csv(patient_csv)

    df_pred = pd.read_csv(pred_csv)
    return df_ecg, df_patients, df_pred

def preprocess_data(df_ecg, df_pred):
    df_ecg['is_within_5_years_of_af'] = df_ecg['is_within_5_years_of_af'].fillna(0)
    df_ecg['path'] = df_ecg['path'].str.replace('/media/data1/ravram/MIMIC-IV/1.0/files/', '/mimic-IV/')
    df_ecg = pd.merge(df_ecg, df_pred, how='inner', left_on='path', right_on='waveform_path')
    df_ecg['pred'] = df_ecg['pred'].str.replace('[', '').str.replace(']', '').astype(float)
    df_ecg = df_ecg[((df_ecg['sinus'] == 1) & ((df_ecg['days_from_af'] < 0) | (df_ecg['days_from_af'].isnull())) & (df_ecg['age_at_ecg'] >= 18) & (df_ecg['days_from_latest'] > 0))]
    return df_ecg

def calculate_metrics(df_ecg):
    auc_val = roc_auc_score(df_ecg['is_within_5_years_of_af'], df_ecg['pred'])
    precision, recall, _ = precision_recall_curve(df_ecg['is_within_5_years_of_af'], df_ecg['pred'])
    ap = average_precision_score(df_ecg['is_within_5_years_of_af'], df_ecg['pred'])
    return auc_val, ap

def calculate_patient_level_data(df_ecg):
    data = df_ecg.copy()  
    grouped_data_0 = calculate_grouped_data(data, 0)
    grouped_data_1 = calculate_grouped_data(data, 1)
    combined_grouped_data = pd.concat([grouped_data_0, grouped_data_1], axis=0).reset_index(drop=True)
    return combined_grouped_data

def plot_metrics(df_ecg, combined_grouped_data):
    pred_cols = ['pred']
    titles = ['ECG-AI']
    suptitle = 'ECG and Patient-Level: Test set'
    os.makedirs('results/plots', exist_ok=True)
    plot_all(df_ecg, combined_grouped_data, set_value=2, pred_cols=pred_cols, titles=titles, suptitle=suptitle, num_bootstraps=1000, output_path='results/plots/metrics_plots.png')

def prepare_kaplan_meier_data(df_ecg):
    data = df_ecg.copy()
    data['days_from_af'] = data['days_from_af'] * (-1)
    data['days_from_af'] = pd.to_timedelta(data['days_from_af'], unit='D')
    data['days_from_latest'] = pd.to_timedelta(data['days_from_latest'], unit='D')
    data['event'] = ~data['days_from_af'].isnull()
    data['duration'] = data[['days_from_af', 'days_from_latest']].min(axis=1).dt.days
    data = data.dropna(subset=['duration', 'event'])
    mimic4_1year = data[(data['days_from_af'] > pd.Timedelta(days=365)) | (data['days_from_af'].isnull())]
    groups = [
        ('MIMIC IV', data),
        ('MIMIC IV 1-year blanking', mimic4_1year)
    ]
    return groups

def plot_kaplan_meier_groups(groups, output_path="results/plots/meier_plot.png"):
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    axes = axes.flatten()
    labels = ['A', 'B']
    times = np.arange(0, 11, 2)

    for i, (title, group) in enumerate(groups):
        high_risk = group[group['pred'] >= 0.12]
        low_risk = group[group['pred'] < 0.12]
        plot_kaplan_meier(axes[i], high_risk, low_risk, title, labels[i], show_legend=(i == 0))
        
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        kmf_high.fit(high_risk['duration'] / 365.25, event_observed=high_risk['event'])
        kmf_low.fit(low_risk['duration'] / 365.25, event_observed=low_risk['event'])
        add_number_at_risk(axes[i], kmf_high, kmf_low, times)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Process and analyze ECG data.")
    parser.add_argument("--ecg_csv", help="Path to the compressed ECG CSV file", required=True)
    parser.add_argument("--patient_csv", help="Path to the compressed patient CSV file", required=True)
    parser.add_argument("--pred_csv", help="Path to the CSV file with predictions", required=True)
    args = parser.parse_args()

    ecg_csv = args.ecg_csv
    patient_csv = args.patient_csv
    pred_csv = args.pred_csv

    df_ecg, df_patients, df_pred = load_data(ecg_csv, patient_csv, pred_csv)
    print(f'Total Number of ECGs: {df_ecg.shape[0]}')
    print(f'Total Number of Patients: {df_patients.shape[0]}')
    print(f'Total Number of Predictions: {df_pred.shape[0]}')

    df_ecg = preprocess_data(df_ecg, df_pred)
    print(df_ecg.describe())

    auc_val, ap = calculate_metrics(df_ecg)
    print(f'AUC: {auc_val}')
    print(f'Precision: {ap}')

    combined_grouped_data = calculate_patient_level_data(df_ecg)
    plot_metrics(df_ecg, combined_grouped_data)

    groups = prepare_kaplan_meier_data(df_ecg)
    plot_kaplan_meier_groups(groups)

if __name__ == '__main__':
    main()
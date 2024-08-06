import os
import pandas as pd
import json
import logging
from datetime import datetime
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
config = {
    'hosp_dir': '/mnt/data1/dcorbin/MimicIV/physionet.org/files/mimiciv/2.2/hosp/',
    'mimic_ecg': '/mnt/data1/ravram/MIMIC-IV/1.0/'
}

def load_data(file_path, compression=None):
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        compression (str): Compression type (e.g., 'gzip'). Default is None.

    Returns:
        pandas.DataFrame: Loaded data as a DataFrame.
    """
    return pd.read_csv(file_path, compression=compression, low_memory=False)

def check_strings(row, string):
    """
    Check if a string is present in any column of a row.

    Args:
        row (pandas.Series): Row of data.
        string (str): String to search for.

    Returns:
        int: 1 if the string is found, 0 otherwise.
    """
    for col in row.index:
        if pd.notna(row[col]) and (string in row[col].lower()):
            return 1
    return 0

def process_ecg_data(ecgs, ecg_measurements):
    """
    Process ECG data by merging and extracting relevant information.

    Args:
        ecgs (pandas.DataFrame): ECG data.
        ecg_measurements (pandas.DataFrame): ECG measurement data.

    Returns:
        pandas.DataFrame: Processed ECG data.
    """
    merged_ecg = pd.merge(ecgs, ecg_measurements, on=['subject_id', 'study_id'])
    report_columns = [col for col in merged_ecg.columns if col.startswith('report_')]
    merged_ecg.loc[:, 'aggregated_report'] = merged_ecg[report_columns].apply(lambda x: '|'.join(x.dropna().astype(str)), axis=1)
    merged_ecg['path'] = merged_ecg['path'].apply(lambda x: os.path.join('/media/data1/ravram/MIMIC-IV/1.0/', x + '.npy'))
    merged_ecg.loc[:, 'afib'] = merged_ecg[merged_ecg.filter(regex='^report_').columns].apply(lambda row: check_strings(row, "atrial fibrillation"), axis=1)
    merged_ecg.loc[:, 'afl'] = merged_ecg[merged_ecg.filter(regex='^report_').columns].apply(lambda row: check_strings(row, "atrial flutter"), axis=1)
    merged_ecg.loc[:, 'sinus'] = merged_ecg[merged_ecg.filter(regex='^report_').columns].apply(lambda row: check_strings(row, "sinus"), axis=1)
    return merged_ecg

def process_patient_data(pid, patient_info, sub_merged_ecg, sub_merged_diagnosis, comorbidities_of_interest, hosp):
    """
    Process patient data and extract relevant information.

    Args:
        pid (int): Patient ID.
        patient_info (pandas.DataFrame): Patient information.
        sub_merged_ecg (pandas.DataFrame): Subset of merged ECG data for the patient.
        sub_merged_diagnosis (pandas.DataFrame): Subset of merged diagnosis data for the patient.
        comorbidities_of_interest (dict): Comorbidities of interest.
        hosp (bool): Indicates if the patient has hospital data.

    Returns:
        dict: Processed patient data.
    """
    patient_row = {}
    patient_row['subject_id'] = pid
    
    if len(patient_info) > 0:
        patient_row['sex'] = patient_info['gender'].values[0]
        patient_row['anchor_age'] = patient_info['anchor_age'].values[0]
        patient_row['anchor_year'] = patient_info['anchor_year'].values[0]
    else:
        logging.warning(f"Patient ID [{pid}] cannot be found in patients.csv. Skipping patient.")
        return None
    
    for comorbidity, data in comorbidities_of_interest.items():
        comorbidity_codes = data['codes']
        comorbidity_name = data['name']
        if hosp:
            patient_row[comorbidity_name] = int(any(code.startswith(tuple(comorbidity_codes)) for code in sub_merged_diagnosis['icd_code']))
        else:
            patient_row[comorbidity_name] = ''
    
    af_codes = comorbidities_of_interest['atrial_fibrillation']['codes'] + comorbidities_of_interest['atrial_flutter']['codes']
    af_diagnosis_mask = sub_merged_diagnosis['icd_code'].str.startswith(tuple(af_codes))
    af_diagnosis = sub_merged_diagnosis[af_diagnosis_mask].copy()
    
    if len(af_diagnosis) > 0:
        af_diagnosis.loc[:, 'admittime'] = pd.to_datetime(af_diagnosis['admittime'])
        patient_row['t_af_hosp_earliest'] = af_diagnosis['admittime'].min()
    else:
        patient_row['t_af_hosp_earliest'] = ''
    
    sub_merged_ecg = sub_merged_ecg.copy()
    sub_merged_ecg.loc[:, 'afib_or_afl'] = ((sub_merged_ecg['afib'] == 1) | (sub_merged_ecg['afl'] == 1)).astype(int)
    sub_merged_ecg_af = sub_merged_ecg[sub_merged_ecg['afib_or_afl'] == 1].copy()
    
    if len(sub_merged_ecg_af) > 0:
        sub_merged_ecg_af.loc[:, 'ecg_time_x'] = pd.to_datetime(sub_merged_ecg_af['ecg_time_x'], format='%Y-%m-%d %H:%M:%S')
        patient_row['t_af_ecg_earliest'] = sub_merged_ecg_af['ecg_time_x'].min()
        patient_row['AFIB'] = 1 if sub_merged_ecg_af['afib'].sum() > 0 else 0
        patient_row['AFL'] = 1 if sub_merged_ecg_af['afl'].sum() > 0 else 0
    else:
        patient_row['t_af_ecg_earliest'] = ''
        patient_row['AFIB'] = 0
        patient_row['AFL'] = 0
    
    patient_row['AF'] = 1 if patient_row['AFIB'] == 1 or patient_row['AFL'] == 1 else 0 if patient_row['AFIB'] == 0 and patient_row['AFL'] == 0 else ''
    
    if len(af_diagnosis) > 0 and len(sub_merged_ecg_af) > 0:
        patient_row['t_af_earliest'] = min([patient_row['t_af_ecg_earliest'], patient_row['t_af_hosp_earliest']])
        patient_row['t_af_earliest_source'] = 'same' if patient_row['t_af_ecg_earliest'] == patient_row['t_af_hosp_earliest'] else 'ecg' if patient_row['t_af_ecg_earliest'] < patient_row['t_af_hosp_earliest'] else 'hosp'
    elif len(af_diagnosis) > 0:
        patient_row['t_af_earliest'] = patient_row['t_af_hosp_earliest']
        patient_row['t_af_earliest_source'] = 'hosp'
    elif len(sub_merged_ecg_af) > 0:
        patient_row['t_af_earliest'] = patient_row['t_af_ecg_earliest']
        patient_row['t_af_earliest_source'] = 'ecg'
    else:
        patient_row['t_af_earliest'] = ''
        patient_row['t_af_earliest_source'] = ''
    
    patient_row['t_latest_ecg'] = pd.to_datetime(sub_merged_ecg['ecg_time_x'].max())
    
    if hosp:
        patient_row['t_latest_hosp'] = pd.to_datetime(sub_merged_diagnosis['dischtime'].max())
        patient_row['t_latest'] = max([patient_row['t_latest_hosp'], patient_row['t_latest_ecg']])
        patient_row['t_latest_source'] = 'hosp' if patient_row['t_latest_hosp'] > patient_row['t_latest_ecg'] else 'ecg' if patient_row['t_latest_hosp'] < patient_row['t_latest_ecg'] else 'same'
    else:
        patient_row['t_latest'] = patient_row['t_latest_ecg']
        patient_row['t_latest_source'] = 'ecg'
    
    patient_row = {key: value.strftime('%Y-%m-%d %H:%M:%S') if isinstance(value, pd.Timestamp) else value for key, value in patient_row.items()}
    
    return patient_row

def process_ecg_rows(sub_merged_ecg, patient_row):
    """
    Process ECG rows and extract relevant information.

    Args:
        sub_merged_ecg (pandas.DataFrame): Subset of merged ECG data for the patient.
        patient_row (dict): Processed patient data.

    Returns:
        pandas.DataFrame: Processed ECG rows.
    """
    ecg_rows = sub_merged_ecg[['subject_id', 'ecg_time_x', 'path', 'afl', 'afib', 'sinus', 'aggregated_report']].copy()
    ecg_rows.loc[:, 'ecg_time_x'] = pd.to_datetime(ecg_rows['ecg_time_x'], format='%Y-%m-%d %H:%M:%S')
    
    if len(patient_row['t_af_earliest']) > 0:
        earliest_date = pd.to_datetime(patient_row['t_af_earliest'])
        ecg_rows.loc[:, 'days_from_af'] = (pd.to_datetime(ecg_rows['ecg_time_x']) - earliest_date).dt.days
        ecg_rows.loc[:, 'is_within_5_years_of_af'] = ((ecg_rows['days_from_af'] >= -(5*365)) & (ecg_rows['days_from_af'] <= 0)).astype(int)
    else:
        ecg_rows.loc[:, 'days_from_af'] = ''
        ecg_rows.loc[:, 'is_within_5_years_of_af'] = ''
    
    ecg_rows.loc[:, 'days_from_latest'] = (pd.to_datetime(patient_row['t_latest']) - pd.to_datetime(ecg_rows['ecg_time_x'])).dt.days
    ecg_rows.loc[:, 'sex'] = patient_row['sex']
    ecg_time = pd.to_datetime(ecg_rows['ecg_time_x'])
    ecg_rows.loc[:, 'age_at_ecg'] = [f.year for f in ecg_time] - patient_row['anchor_year'] + patient_row['anchor_age']
    
    return ecg_rows

def main():
    """
    Main function to process the data and generate output files.
    """
    # Read data files
    admissions = load_data(os.path.join(config['hosp_dir'], 'admissions.csv.gz'), compression='gzip')
    diagnoses = load_data(os.path.join(config['hosp_dir'], 'diagnoses_icd.csv'))
    patients = load_data(os.path.join(config['hosp_dir'], 'patients.csv.gz'), compression='gzip')
    icd_code = load_data(os.path.join(config['hosp_dir'], 'd_icd_diagnoses.csv.gz'), compression='gzip')
    ecgs = load_data(os.path.join(config['mimic_ecg'], 'record_list.csv'))
    ecg_measurements = load_data(os.path.join(config['mimic_ecg'], 'machine_measurements.csv'))
    
    merged_ecg = process_ecg_data(ecgs, ecg_measurements)
    merged_diagnosis = pd.merge(admissions, diagnoses, on=['subject_id', 'hadm_id'])
    
    with open('icd_codes.json') as json_data:
        comorbidities_of_interest = json.load(json_data)
    
    final_ecg_rows = pd.DataFrame()
    patient_rows_df = pd.DataFrame()
    
    for pid in tqdm(merged_ecg['subject_id'].unique()[:10]):
        patient_info = patients[patients['subject_id'] == pid]
        sub_merged_ecg = merged_ecg[merged_ecg['subject_id'] == pid]
        sub_merged_diagnosis = merged_diagnosis[merged_diagnosis['subject_id'] == pid]
        hosp = merged_diagnosis['subject_id'].isin([pid]).any()
        patient_row = process_patient_data(pid, patient_info, sub_merged_ecg, sub_merged_diagnosis, comorbidities_of_interest, hosp)
        
        if patient_row is not None:
            ecg_rows = process_ecg_rows(sub_merged_ecg, patient_row)
            final_ecg_rows = pd.concat([final_ecg_rows, ecg_rows], ignore_index=True)
            patient_rows_df = pd.concat([patient_rows_df, pd.DataFrame([patient_row])], ignore_index=True)
    
    final_ecg_rows.to_csv('final_ecg_rows.csv', index=False)
    patient_rows_df.to_csv('patient_rows.csv', index=False)

if __name__ == '__main__':
    main()
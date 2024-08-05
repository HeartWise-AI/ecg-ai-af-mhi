import pandas as pd 
import json 
import numpy as np 
from datetime import datetime
from tqdm import tqdm 

### Read data files
admissions = pd.read_csv('/media/data1/dcorbin/MimicIV/physionet.org/files/mimiciv/2.2/hosp/admissions.csv.gz',compression='gzip')
diagnoses = pd.read_csv('/media/data1/dcorbin/MimicIV/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv')
patients = pd.read_csv('/media/data1/dcorbin/MimicIV/physionet.org/files/mimiciv/2.2/hosp/patients.csv.gz',compression='gzip')
icd_code = pd.read_csv('/media/data1/dcorbin/MimicIV/physionet.org/files/mimiciv/2.2/hosp/d_icd_diagnoses.csv.gz', compression='gzip')
ecgs = pd.read_csv('/media/data1/ravram/MIMIC-IV/1.0/record_list.csv')
ecg_measurements = pd.read_csv('/media/data1/ravram/MIMIC-IV/1.0/machine_measurements.csv',low_memory=False)


##

def check_strings(row, string):
    for col in row.index:
        if pd.notna(row[col]) and (string in row[col].lower()):
            return 1
    return 0

merged_ecg = pd.merge(ecgs, ecg_measurements, on=['subject_id', 'study_id'])
report_columns = [col for col in merged_ecg.columns if col.startswith('report_')]
merged_ecg.loc[:,'aggregated_report'] = merged_ecg[report_columns].apply(lambda x: '|'.join(x.dropna().astype(str)), axis=1)

merged_diagnosis = pd.merge(admissions, diagnoses, on=['subject_id', 'hadm_id'])

merged_ecg['path'] = merged_ecg['path'].apply(lambda x: '/media/data1/ravram/MIMIC-IV/1.0/' + x + '.npy')
merged_ecg.loc[:, 'afib'] = merged_ecg[merged_ecg.filter(regex='^report_').columns].apply(lambda row: check_strings(row, "atrial fibrillation"), axis=1)
merged_ecg.loc[:, 'afl'] = merged_ecg[merged_ecg.filter(regex='^report_').columns].apply(lambda row: check_strings(row, "atrial flutter"), axis=1)
merged_ecg.loc[:, 'sinus'] = merged_ecg[merged_ecg.filter(regex='^report_').columns].apply(lambda row: check_strings(row, "sinus"), axis=1)

with open('icd_codes.json') as json_data:
    coi = json.load(json_data)
    json_data.close()
    
final_ecg_rows = pd.DataFrame()
patient_rows_df = pd.DataFrame()
problematic_pids = []
for pid in tqdm(merged_ecg['subject_id'].unique()):
    
    patient_row = {}
    patient_info = patients[patients['subject_id'] == pid]
    sub_merged_ecg = merged_ecg[merged_ecg['subject_id']==pid]
    sub_merged_diagnosis = merged_diagnosis[merged_diagnosis['subject_id']==pid]

    patient_row['subject_id'] = pid
    if len(patient_info)>0:
        patient_row['sex'] = patient_info['gender'].values[0]
        patient_row['anchor_age'] = patient_info['anchor_age'].values[0]   
        patient_row['anchor_year'] = patient_info['anchor_year'].values[0]
    else:
        problematic_pids.append(pid)
        continue
    
    if merged_diagnosis['subject_id'].isin([pid]).any():
        hosp = True
    else:
        hosp = False
        
    for comorbidity in coi.keys():
        comorbidity_codes = coi[comorbidity]['codes']
        comorbidity_name = coi[comorbidity]['name']
        if hosp:
            patient_row[comorbidity_name] = int(any(code.startswith(tuple(comorbidity_codes)) for code in sub_merged_diagnosis['icd_code']))
        else:
            patient_row[comorbidity_name] = ''
        
    af_codes = coi['atrial_fibrillation']['codes']+coi['atrial_flutter']['codes']
    af_diagnosis_mask = sub_merged_diagnosis['icd_code'].str.startswith(tuple(af_codes))
    af_diagnosis = sub_merged_diagnosis[af_diagnosis_mask]
    af_diagnosis = af_diagnosis.copy()
    
    if len(af_diagnosis)>0:
        af_diagnosis.loc[:, 'admittime'] = pd.to_datetime(af_diagnosis['admittime'])
        patient_row['t_af_hosp_earliest'] = af_diagnosis['admittime'].min()
    else:
        patient_row['t_af_hosp_earliest'] = ''
    
    
    sub_merged_ecg = sub_merged_ecg.copy()
    report_columns = sub_merged_ecg.filter(regex='^report_').columns
    sub_merged_ecg.loc[:, 'afib_or_afl'] = ((sub_merged_ecg['afib'] == 1) | (sub_merged_ecg['afl'] == 1)).astype(int)
    sub_merged_ecg_af = sub_merged_ecg[sub_merged_ecg['afib_or_afl'] == 1]
    sub_merged_ecg_af = sub_merged_ecg_af.copy()

    if len(sub_merged_ecg_af)>0:
        sub_merged_ecg_af.loc[:, 'ecg_time_x'] = pd.to_datetime(sub_merged_ecg_af['ecg_time_x'], format='%Y-%m-%d %H:%M:%S')
        patient_row['t_af_ecg_earliest'] = sub_merged_ecg_af['ecg_time_x'].min()
        if sub_merged_ecg_af['afib'].sum()>0:
            patient_row['AFIB'] = 1
        if sub_merged_ecg_af['afl'].sum()>0:
            patient_row['AFL'] = 1
    else:
        patient_row['t_af_ecg_earliest'] = ''
        
    if patient_row['AFIB'] == 1 or patient_row['AFL'] == 1:
        patient_row['AF'] = 1
    elif patient_row['AFIB'] == 0 and patient_row['AFL'] == 0:
        patient_row['AF'] = 0
    else:
        patient_row['AF'] = ''
        
    if len(af_diagnosis)>0 and len(sub_merged_ecg_af)>0:
        patient_row['t_af_earliest'] = min([patient_row['t_af_ecg_earliest'], patient_row['t_af_hosp_earliest']])
        if patient_row['t_af_ecg_earliest'] == patient_row['t_af_hosp_earliest']:
            patient_row['t_af_earliest_source'] = 'same'
        elif patient_row['t_af_ecg_earliest'] < patient_row['t_af_hosp_earliest']:
            patient_row['t_af_earliest_source'] = 'ecg'  
        else:
            patient_row['t_af_earliest_source'] = 'hosp'                
    elif len(af_diagnosis)>0:
        patient_row['t_af_earliest'] = patient_row['t_af_hosp_earliest']
        patient_row['t_af_earliest_source'] = 'hosp'
    elif len(sub_merged_ecg_af)>0:
        patient_row['t_af_earliest'] = patient_row['t_af_ecg_earliest']
        patient_row['t_af_earliest_source'] = 'ecg'
    else:
        patient_row['t_af_earliest'] = ''
        patient_row['t_af_earliest_source'] = ''
    
    patient_row['t_latest_ecg'] = pd.to_datetime(sub_merged_ecg['ecg_time_x'].max())
    
    if hosp:
        patient_row['t_latest_hosp'] = pd.to_datetime(sub_merged_diagnosis['dischtime'].max())
        patient_row['t_latest'] = max([patient_row['t_latest_hosp'],patient_row['t_latest_ecg']])
        if patient_row['t_latest_hosp'] > patient_row['t_latest_ecg']:
            patient_row['t_latest_source'] = 'hosp'
        elif patient_row['t_latest_hosp'] < patient_row['t_latest_ecg']:
            patient_row['t_latest_source'] = 'ecg'
        else:  
            patient_row['t_latest_source'] = 'same'
    else:
        patient_row['t_latest'] = patient_row['t_latest_ecg']
        patient_row['t_latest_source'] = 'ecg'

    patient_row = {key: value.strftime('%Y-%m-%d %H:%M:%S') if isinstance(value, pd.Timestamp) else value for key, value in patient_row.items()}
    
    ecg_rows = sub_merged_ecg[['subject_id','ecg_time_x','path','afl','afib','sinus','aggregated_report']]
    ecg_rows = ecg_rows.copy()
    ecg_rows.loc[:, 'ecg_time_x'] = pd.to_datetime(ecg_rows['ecg_time_x'], format='%Y-%m-%d %H:%M:%S')
    
    if len(patient_row['t_af_earliest'])>0:
        earliest_date = pd.to_datetime(patient_row['t_af_earliest'])
        ecg_rows.loc[:, 'dt_af'] = (pd.to_datetime(ecg_rows['ecg_time_x']) - earliest_date).dt.days
        ecg_rows.loc[:, 'y_5y'] = ((ecg_rows['dt_af'] >= -(5*365)) & (ecg_rows['dt_af'] <= 0)).astype(int)
    else:
        ecg_rows.loc[:,'dt_af'] = ''
        ecg_rows.loc[:,'y_5y'] = ''
        
    ecg_rows.loc[:, 'dt_max'] = (pd.to_datetime(patient_row['t_latest']) - pd.to_datetime(ecg_rows['ecg_time_x'])).dt.days
    ecg_rows.loc[:, 'sex'] = patient_row['sex']
    ecg_time = pd.to_datetime(ecg_rows['ecg_time_x'])
    ecg_rows.loc[:, 'age_at_ecg'] = [f.year for f in ecg_time] - patient_row['anchor_year'] + patient_row['anchor_age']
    
    final_ecg_rows = pd.concat([final_ecg_rows, ecg_rows], ignore_index=True)
    patient_rows_df = pd.concat([patient_rows_df, pd.DataFrame([patient_row])], ignore_index=True)
    
final_ecg_rows.to_csv('final_ecg_rows_2.csv', index=False)
patient_rows_df.to_csv('patient_rows_2.csv', index=False)
    

print(problematic_pids)

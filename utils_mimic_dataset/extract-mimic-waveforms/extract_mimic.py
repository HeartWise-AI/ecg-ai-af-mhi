import sys
import wfdb
import pandas as pd
import numpy as np 
from tqdm import tqdm
import json

with open('mimic_paths.json') as f:
    params = json.load(f)

def read_params_path(params_path):
    if params_path.endswith('gz'):
        return pd.read_csv(params_path, compression='gzip')
    else:
        return pd.read_csv(params_path)

records = read_params_path(params['records'])
metadata = read_params_path(params['metadata'])
patients = read_params_path(params['patients'])
ecg_features = ['rr_interval', 'p_onset', 'p_end', 'qrs_onset', 'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis']

def get_ecg(idx,output_file):
    
    ecg_infos = records.iloc[idx]
    rec_path = f'{params['wf_path']}{ecg_infos["path"]}' 
    infos = rec_path.split('/')
    subject_id = infos[8][1:]
    study_id = infos[9][1:]
    
    ecg_time = pd.to_datetime(ecg_infos['ecg_time'])
    patient_info = patients[patients['subject_id']==int(subject_id)]
    if len(patient_info)>0:
        age_at_ecg = ecg_time.year - patient_info.loc[patient_info.index[0], 'anchor_year'] + patient_info.loc[patient_info.index[0], 'anchor_age']
        gender = patient_info.loc[patient_info.index[0], 'gender']
    else:
        age_at_ecg = 'missing'
        gender = 'missing'
    
    rd_record = wfdb.rdrecord(rec_path) 
    ecg_signal_uV = rd_record.p_signal*1000
    
    meta = metadata[(metadata['subject_id']==int(subject_id)) & (metadata['study_id']==int(study_id))]
    reports = meta.filter(like='report_').values.ravel()
    concatenated_reports = ';'.join(str(x) for x in reports if pd.notna(x))
    output_str = f'{gender},{age_at_ecg},{concatenated_reports},{params['output_dir']}{ecg_infos["path"]}.npy'
    
    for feature in ecg_features:
        if feature in meta.columns:
            output_str += f',{meta[feature].values[0]}'
        else:
            output_str += ','
    
    with open(output_file,'a') as of:
        of.write(output_str+'\n')
        
    np.save(f'{params['output_dir']}{ecg_infos["path"]}.npy', ecg_signal_uV)
    
if __name__ == '__main__':
    output_file = sys.argv[1]
    print(output_file)
    for idx in tqdm(range(len(records))):
        get_ecg(idx, output_file)

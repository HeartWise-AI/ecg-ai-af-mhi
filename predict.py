import xmltodict
import base64
import struct
import numpy as np 
from glob import glob
import argparse 
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from huggingface_hub import snapshot_download

lead_order = [
        "I",
        "II",
        "III",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6"
    ]


def decode_ekg_muse_to_array(raw_wave, downsample=1):
    """
    Ingest the base64 encoded waveforms and transform to numeric
    downsample: 0.5 takes every other value in the array. Muse samples at 500/s and the sample model requires 250/s. So take every other.
    """
    try:
        dwnsmpl = int(1 // downsample)
    except ZeroDivisionError:
        print("You must downsample by more than 0")
    # covert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, "utf-8"))

    # unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = "".join([char * int(len(arr) / 2) for char in "h"])
    byte_array = struct.unpack(unpack_symbols, arr)
    return np.array(byte_array)[::dwnsmpl]

def extract_wf_as_npy(xml_folder):
    lead_data = dict.fromkeys(lead_order)

    all_ekg_arrays = []
    print(xml_folder)
    for file_xml in glob(f"{xml_folder}/*.xml"):
        with open(file_xml, encoding='ISO-8859-1') as xml:
            ECG_data_nested = xmltodict.parse(xml.read())
            for lead in ECG_data_nested["RestingECG"]["Waveform"]:
                for leadid in range(len(lead["LeadData"])):
                    sample_length = len(decode_ekg_muse_to_array(lead["LeadData"][leadid]["WaveFormData"]))
                    if sample_length == 5000:  # check if the length is 5000
                        shape_ecg = 5000  # keep in mind the original shape to add it to the output parquet
                        lead_data[
                            lead["LeadData"][leadid]["LeadID"]
                        ] = decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=0.5
                        )
                    elif sample_length == 2500:  # check if the length is 2500
                        shape_ecg = 2500  # keep in mind the original shape to add it to the output parquet
                        lead_data[
                            lead["LeadData"][leadid]["LeadID"]
                        ] = decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=1
                        )
                    else:
                        continue
            # convert the leads that are measured into the calculated leads
            lead_data["III"] = np.array(lead_data["II"]) - np.array(lead_data["I"])
            lead_data["aVR"] = -(np.array(lead_data["I"]) + np.array(lead_data["II"])) / 2
            lead_data["aVF"] = (np.array(lead_data["II"]) + np.array(lead_data["III"])) / 2
            lead_data["aVL"] = (np.array(lead_data["I"]) - np.array(lead_data["III"])) / 2

            lead_data = {k: lead_data[k] for k in lead_order}

            temp = []
            for key, value in lead_data.items():
                temp.append(value)

            # transpose to be [time, leads, ]
            ekg_array = np.array(temp).T
            # expand dims to [time, leads, 1]
            ekg_array = np.expand_dims(ekg_array, axis=0)
            all_ekg_arrays.append(ekg_array)
            
    final_ekg_array = np.concatenate(all_ekg_arrays, axis=0)
    return final_ekg_array

            
def make_inference(xml_dir):
    # Importing the model
    print('Loading the model')
    model_name='best_model.h5'
    model_path='./'+model_name
    model = load_model(model_path, custom_objects={'Addons>F1Score': tfa.metrics.F1Score})


    # Load the data
    print('Loading the data')
    X=extract_wf_as_npy(xml_dir)

    print(X.shape)
    
    # Standard scaler
    print('Applying standard scaler')
    mean = -0.75227547
    std = 41.20366
    X_std = (X - mean) / std
    X_std = X_std.astype(np.float16)

    # Model predictions
    y_pred = model(X_std)
    return(y_pred)

    
def get_arguments():
    """
    Generate the arguments using CLI
    """
    parser = argparse.ArgumentParser(description="Get argument", add_help=False)
    parser.add_argument(
        "--xml_dir",
        metavar="xml_dir",
        type=str,
        help="Enter path to xml",
    )
    return parser

if __name__ == '__main__':
    snapshot_download(repo_id="heartwise/ecgAI_AF_MHI", local_dir="./", token='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')
    
    print(make_inference('/xml-data/'))
    
    
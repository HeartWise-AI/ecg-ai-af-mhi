import xmltodict
import base64
import struct
import numpy as np
from glob import glob

lead_order = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def decode_ekg_muse_to_array(raw_wave, downsample=1):
    """
    Ingest the base64 encoded waveforms and transform to numeric.

    Args:
        raw_wave (str): Base64 encoded waveform data.
        downsample (float): Downsampling factor. Default is 1.

    Returns:
        numpy.ndarray: Decoded and downsampled waveform data.
    """
    try:
        dwnsmpl = int(1 // downsample)
    except ZeroDivisionError:
        print("You must downsample by more than 0")
    # Convert the waveform from base64 to byte array
    arr = base64.b64decode(bytes(raw_wave, "utf-8"))

    # Unpack every 2 bytes, little endian (16 bit encoding)
    unpack_symbols = "".join([char * int(len(arr) / 2) for char in "h"])
    byte_array = struct.unpack(unpack_symbols, arr)
    return np.array(byte_array)[::dwnsmpl]

def xml_waveform_extract(xml_file):
    """
    Extract waveform data from an XML file.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        numpy.ndarray: Extracted waveform data as a 3D array [time, leads, 1].
    """
    lead_data = dict.fromkeys(lead_order)
    with open(xml_file, encoding="ISO-8859-1") as xml:
        ECG_data_nested = xmltodict.parse(xml.read())
        for lead in ECG_data_nested["RestingECG"]["Waveform"]:
            for leadid in range(len(lead["LeadData"])):
                sample_length = len(
                    decode_ekg_muse_to_array(lead["LeadData"][leadid]["WaveFormData"])
                )
                if sample_length == 5000:  # Check if the length is 5000
                    shape_ecg = 5000  # Keep in mind the original shape to add it to the output parquet
                    lead_data[lead["LeadData"][leadid]["LeadID"]] = (
                        decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=0.5
                        )
                    )
                elif sample_length == 2500:  # Check if the length is 2500
                    shape_ecg = 2500  # Keep in mind the original shape to add it to the output parquet
                    lead_data[lead["LeadData"][leadid]["LeadID"]] = (
                        decode_ekg_muse_to_array(
                            lead["LeadData"][leadid]["WaveFormData"], downsample=1
                        )
                    )
                else:
                    continue
        # Convert the leads that are measured into the calculated leads
        lead_data["III"] = np.array(lead_data["II"]) - np.array(lead_data["I"])
        lead_data["aVR"] = -(np.array(lead_data["I"]) + np.array(lead_data["II"])) / 2
        lead_data["aVF"] = (np.array(lead_data["II"]) + np.array(lead_data["III"])) / 2
        lead_data["aVL"] = (np.array(lead_data["I"]) - np.array(lead_data["III"])) / 2

        lead_data = {k: lead_data[k] for k in lead_order}

        temp = []
        for key, value in lead_data.items():
            temp.append(value)

        # Transpose to be [time, leads]
        ekg_array = np.array(temp).T
        # Expand dimensions to [time, leads, 1]
        ekg_array = np.expand_dims(ekg_array, axis=0)

    return ekg_array
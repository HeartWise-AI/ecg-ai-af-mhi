# ECG AI Model Inference

This repository contains Python code for loading ECG (electrocardiogram) data from XML files, preprocessing it, and running inference using a deep learning model. The main functionality is to decode encoded ECG waveforms from XML, transform them to NumPy arrays, and apply a trained model to predict outcomes. The results are then saved as a CSV file. 

Model Outcome: Incident atrial fibrillation or atrial flutter at 5 years at MHI (0/1)

*Atrial fibrillation (AF) and atrial flutter (AFL) are abnormal heart rhythms originating in the upper chambers of the heart (atria). They cause the atria to beat irregularly and often faster than normal, disrupting the heart's coordinated contraction. Detecting AF and AFL is important because they can lead to serious complications such as stroke, heart failure, and reduced quality of life. Early detection allows for timely treatment, which may include medications, procedures, or other interventions to restore normal heart rhythm and prevent complications. ECG-based AI models, like the one in this repository, can assist in the automated detection of these arrhythmias, enabling prompt diagnosis and management.*

## Preparing MIMIC-Ext dataset

*The MIMIC-IV-ECG dataset, a subset of the MIMIC-IV Clinical Database, contains approximately 800,000 diagnostic ECGs from nearly 160,000 unique subjects, along with associated machine measurements and cardiologist reports. Validating an AI model on this external dataset offers several benefits, including assessing generalizability, evaluating robustness to different patient populations and clinical settings, reducing biases, enabling benchmarking, and exploring the model's potential for clinical decision support. By validating the model on MIMIC-IV-ECG, researchers can ensure its reliability and effectiveness in real-world applications, ultimately contributing to the development of more advanced AI tools in healthcare.*

### Extracting Waveform Signal

``` python utils_mimic_dataset/extract-mimic-data/extract_mimic.py --records path/to/records.csv --metadata path/to/machine_measurements.csv --patients path/to/patients.csv.gz --wf_path path/to/wf/ --output_dir path/to/output_dir --output_file path/to/output_file ```

### Creating Ground-Truth Table for Atrial Fibrillation/Atrial Flutter

``` python utils_mimic_dataset/extract-mimic-data/create_tables.py ```

### Find Scaling Factors for MIMIC dataset 

``` python utils_mimic_dataset/extract-mimic-data/scale_mimic_waveforms.py --data_dir path/to/root/mimic-npy ```

## Inference Setup

### Huggingface gated access
The model weights are public but their access is gated. You must first login to huggingface and request access to the [weights](https://huggingface.co/heartwise/ecgAI_AF_MHI). 
You will be granted access immediately. 

### Huggingface authentification 
Because the files are gated, you need to be authenticated to [huggingface](https://huggingface.co/) to access the model weights. This is required to run the inference. 
From your huggingface account, obtain a [read-access token](https://huggingface.co/docs/hub/en/security-tokens) in `Settings > Access Token > + Create New Token`. 

Use a .env file to load your access token
```
cp .env.example .env
```
Change `<your-access-token>` with your hf access token

## Inference Using Docker

The recommended way to run this code is to use docker. The model makes predictions from MUSE XML ECG files. 
You need to first prepare your input files by putting them in the `./xml-data` directory

### Build docker image 
```
docker build -t ecg-ai-af-mhi .
```

### Run docker
```
docker run \
  --gpus all \
  --env-file .env \
  -v ./xml-data/:/xml-data/ \
  -v ./weights/:/weights/ \
  -v ./results/:/results/ \
  ecg-ai-af-mhi
```

## Run the code 

It is strongly recommended to use the docker steps but you can also try to run the code directly in the terminal. 
However, the versions of cuda and tensorflow need to match and this option currently doesn't work on macos. 

Install the packages in a virtual environment using `pip`:

```shell
conda create -n ai-env python=3.8
conda activate ai-env
pip install -r requirements.txt
```

Make sure the .env file is sourced to have access to the HF_TOKEN environment variable 
```
source .env
```

Run the code using the parameters json file. This file points to the directories used by the code (ie, xml-data, weights, results)
```
python predict.py --config params.json --MHI|--MIMIC
```

### Results 

The results can be found in `results/results.csv`
Validation metrics can be obtained by running: ``` python utils_mimic_dataset/external_validation_statistics.py --ecg_csv path/to/final_ecgs_mimic.csv --patient_csv path/to/patient_mimic.csv --pred_csv path/to/predictions.csv ```
 

## License
[...]

## Citation
If you use this code for academic research, please cite it as follows:

```
@software{your_name_year,
  author = {[Your Name]},
  title = {{ECG AI Model Inference}},
  url = {https://github.com/HeartWise-AI/ecg-ai-af-mhi},
  year = {2024}
}
```


## Contact
For any queries or concerns, please open an issue on the GitHub repository or contact the authors directly.

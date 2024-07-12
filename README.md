# ECG AI Model Inference

This repository contains Python code for loading ECG (electrocardiogram) data from XML files, preprocessing it, and running inference using a deep learning model. The main functionality is to decode encoded ECG waveforms from XML, transform them to NumPy arrays, and apply a trained model to predict outcomes. The results are then saved as a CSV file. 

Model Outcome: Incident atrial fibrillation or atrial flutter at 5 years at MHI (0/1)

## Setup

### Huggingface gated access
The model weights are public but their access is gated. You must first login to huggingface and request access to the [weights](https://huggingface.co/heartwise/ecgAI_AF_MHI). 
You will be granted access immediately. 

### Huggingface authentification 
Because the files are gated, you need to be authenticated to [huggingface](https://huggingface.co/) to access the model weights. This is required to run the inference. 
From your huggingface account, obtain a read-access token in `Settings > Access Token > + Create New Token`. 

Use a .env file to load your access token
```
cp .env.example .env
```
Change `<your-access-token>` with your hf access token

## Using Docker

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
python predict.py --config params.json
```

### Results 
The results can be found in `results/results.csv`

## License
[...]

## Citation
If you use this code for academic research, please cite it as follows:

```
@software{your_name_year,
  author = {[Your Name]},
  title = {{ECG AI Model Inference}},
  url = {https://github.com/[Your GitHub Username]/[Your Repository Name]},
  year = {[Year]}
}
```

Remember to include the LICENSE file and replace [Your Name], [Year], [Your GitHub Username], and [Your Repository Name] with your information.


## Contact
For any queries or concerns, please open an issue on the GitHub repository or contact the authors directly.

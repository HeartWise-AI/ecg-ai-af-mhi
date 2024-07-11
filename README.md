# ECG AI Model Inference

This repository contains Python code for loading ECG (electrocardiogram) data from XML files, preprocessing it, and running inference using a deep learning model. The main functionality is to decode encoded ECG waveforms from XML, transform them to NumPy arrays, and apply a trained model to predict outcomes. The results are then saved as a CSV file. 

## Setup

### Huggingface authentification 
You need to be authenticated to [huggingface](https://huggingface.co/) to access the model weights. This is required to run the inference. 
Login to your huggingface account and obtain a read-access token in `Settings > Access Token > + Create New Token`. 

Use a .env file to load your access token
```
cp .env.example .env
```
Change `<your-access-token>` with your hf access token

## Using Docker

The recommended way to run this code is to use docker. 

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

The results can be found in ./results/results.csv


## Run the code 

If you prefer running the code directly, follow these steps:

You can install the packages in a virtual environment using `pip`:

```shell
conda create -n ai-env python=3.8
conda activate ai-env
pip install -r requirements.txt
```

A config file is used to indicate the different folders. Please specify the: 

* `model_path` : where the model weights will be downloaded 

* `xml_dir` : where the xml files are stored 

* `output_dir` : where the results.csv will be saved

It is suggested to create those directories inside the repo: 
```
mkdir xml-data
mkdir weights
mkdir results
```


```
python predict.py --config params.json
```


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

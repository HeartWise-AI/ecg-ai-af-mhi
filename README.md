# ECG AI Model Inference

This repository contains Python code for loading ECG (electrocardiogram) data from XML files, preprocessing it, and running inference using a deep learning model. The main functionality is to decode encoded ECG waveforms from XML, transform them to NumPy arrays, and apply a trained model to predict outcomes. The results are then saved as a CSV file. 

## Prerequisites

Before running this code, ensure you have the following packages installed:

- `xmltodict`: For parsing XML files.
- `numpy`: For numerical operations on arrays.
- `tensorflow` (with GPU support if available): For loading and using the deep learning model.
- `tensorflow_addons`: For additional TensorFlow metrics.
- `huggingface_hub`: For downloading models from Hugging Face Hub.

You can install these packages in a virtual environment using `pip`:

```shell
conda create -n ai-env python=3.8 
pip install -r requirements.txt
```

## How to Run the Code
To run the inference with the ECG AI model, follow these steps:

### Clone this repository.
Place your XML files containing the ECG data in a directory (e.g., ./xml-data/).
Use the following command to run the script. Make sure to replace /path/to/xml_directory with your directory path containing the XML files:

### Huggingface authentification 
Executing the code will pull the model weights from huggingface. You need to be authenticated. Login to your huggingface account and obtain a read-access token. 
Run `export HF_TOKEN=<your-access-token-here>`

### Run the code 


The inference results will be printed to the console.

## Downloading the Model
The code is configured to download a pre-trained model from the Hugging Face Hub using the snapshot_download function. Ensure you have the proper authentication token to download the model.

## License
This project is licensed under the MIT License.

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

## Acknowledgments
[...Any special acknowledgments or credits...]

## Contact
For any queries or concerns, please open an issue on the GitHub repository or contact the authors directly.

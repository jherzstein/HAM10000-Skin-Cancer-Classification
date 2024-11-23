import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate the Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset and download path
dataset = 'surajghuwalewala/ham1000-segmentation-and-classification'
download_path = './data/'

# Create the directory if it doesn't exist
os.makedirs(download_path, exist_ok=True)

# Download the dataset
api.dataset_download_files(dataset, path=download_path, unzip=True)

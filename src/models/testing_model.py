"""
Module Name: testing_model.py

This module provides a pipeline for testing a model using MLflow integration.
"""
import configparser
import os
import sys

from codecarbon import EmissionsTracker
import mlflow
import mlflow.pytorch
import torch

from data import load_data
from modelling import evaluation_mflow

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
sys.path.insert(1, os.path.join(root_dir, 'src/vision'))

import utils


# Download preprocessed data
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
DATA_FOLDER = os.path.join(root_dir, 'data/processed')
_, _, testing_dataset = load_data(DATA_FOLDER)

# Set environment variables
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RodrigoBonferroni/" \
                                    "taed2-PedestrianDetection.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "claudiamur"
config = configparser.ConfigParser()
config.read(os.path.join(root_dir,'config.ini'))
os.environ["MLFLOW_TRACKING_PASSWORD"] = config['Credentials']['mlflow_password']
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


# Define the data loader
data_loader_test = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=True, num_workers=0,
    collate_fn=utils.collate_fn)

# Start tracking emissions for training
tracker = EmissionsTracker(
    output_file=os.path.join(root_dir, 'metrics/emissions.csv'),
    on_csv_write='append',
)
tracker.start()

mlflow.set_experiment("Pipeline retraining")

# load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

final_model = torch.load(os.path.join(root_dir, 'models/baseline.pth'), map_location=device)

evaluation_mflow('test', data_loader_test, final_model, device,
                 emissions_file=os.path.join(root_dir, 'metrics/emissions.csv'))
tracker.stop()

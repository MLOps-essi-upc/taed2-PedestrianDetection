from codecarbon import EmissionsTracker
import mlflow
import mlflow.pytorch
from mlflow import log_metric, log_param, log_params, log_artifacts
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import configparser
from data import load_data
from modelling import evaluation_mflow, train_mlflow
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
sys.path.insert(1, os.path.join(root_dir, 'src/data'))
sys.path.insert(1, os.path.join(root_dir, 'src/features'))

#from pedestrian_dataset_class import PedestrianDataset  
from engine import train_one_epoch, evaluate
import utils 


# Download preprocessed data
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
DATA_FOLDER = os.path.join(root_dir, 'data/processed')
_, _, testing_dataset = load_data(DATA_FOLDER)


# Set environment variables
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RodrigoBonferroni/taed2-PedestrianDetection.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "claudiamur"
config = configparser.ConfigParser()
config.read(os.path.join(root_dir,'config.ini'))
os.environ["MLFLOW_TRACKING_PASSWORD"] = config['Credentials']['mlflow_password']
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


# Define the data loader
data_loader_test = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)


# Start tracking emissions for training
tracker = EmissionsTracker(
    output_file=os.path.join(root_dir, 'metrics/emissions_retraining.csv'),
    on_csv_write='append',
)
tracker.start()

mlflow.set_experiment("Pipeline retraining")

# load best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

final_model = torch.load(os.path.join(root_dir, 'models/baseline.pth'), map_location=device)


evaluation_mflow('test', data_loader_test, final_model, device)
tracker.stop()

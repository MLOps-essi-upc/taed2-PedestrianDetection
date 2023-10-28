from codecarbon import EmissionsTracker
from src.models.data import load_data
# Import the required class
from src.data.pedestrian_dataset_class import PedestrianDataset
import mlflow
import mlflow.pytorch
from mlflow import log_metric, log_param, log_params, log_artifacts
import torch
import utils
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import os
import getpass
from src.features.engine import train_one_epoch, evaluate
from src.models.modelling import evaluation_mflow, train_mlflow
import configparser


# Download preprocessed data
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
DATA_FOLDER = os.path.join(root_dir, 'processed_data')
_, _, testing_dataset = load_data(DATA_FOLDER)


# Set environment variables
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RodrigoBonferroni/taed2-PedestrianDetection.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "claudiamur"
config = configparser.ConfigParser()
config.read('config.ini')
password = config['Credentials']['mlflow_password']
os.environ["MLFLOW_TRACKING_PASSWORD"] = password
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

mlflow.set_experiment("Pipeline: Fine tunning")

# load best model
final_model = torch.load('baseline.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

evaluation_mflow('test', data_loader_test, final_model, device)
tracker.stop()

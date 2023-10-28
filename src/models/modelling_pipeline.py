from codecarbon import EmissionsTracker
from src.models.data import load_data
from src.data.pedestrian_dataset_class import PedestrianDataset  # Import the required class
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



# Parameters (in params.yaml)
batch_size_train = 2
hidden_layer = 256  # for mask predictor
training_epochs = 3


# Download preprocessed data
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
DATA_FOLDER = os.path.join(root_dir, 'processed_data')
training_dataset, validation_dataset, _ = load_data(DATA_FOLDER)


# Set environment variables
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RodrigoBonferroni/taed2-PedestrianDetection.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "claudiamur"
config = configparser.ConfigParser()
config.read('config.ini')
password = config['Credentials']['mlflow_password']
os.environ["MLFLOW_TRACKING_PASSWORD"] = password
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


# Define the data loaders
data_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    validation_dataset, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)


# Start tracking emissions for training
tracker = EmissionsTracker(
    output_file=os.path.join(root_dir, 'metrics/emissions_retraining.csv'),
    on_csv_write='append',
)
tracker.start()

# Import pre-trained model mask rcnn resnet 50
# load an instance segmentation model pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)

# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
num_classes = 2
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# Fine tunning experiment
mlflow.set_experiment("Pipeline: Fine tunning")
tracker.start()
train_mlflow(model, data_loader, data_loader_val, num_epochs=training_epochs, hidden_layer=hidden_layer,
             batch_size_train=batch_size_train, name="baseline", device=device)
tracker.stop()

torch.save(model, os.path.join(root_dir, 'models/baseline.pth')) # es sobreescriu


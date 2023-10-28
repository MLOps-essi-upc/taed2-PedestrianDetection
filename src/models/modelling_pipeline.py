from codecarbon import EmissionsTracker
from data import load_data
from PedestrianDatasetClass import PedestrianDataset  # Import the required class
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
from engine import train_one_epoch, evaluate
from modelling import evaluation_mflow, train_mlflow

# Download preprocessed data
DATA_FOLDER = ""
training_dataset, validation_dataset, testing_dataset = load_data(DATA_FOLDER)


# set environment variables
password = getpass.getpass(prompt='Enter your password: ')  # potser que estingui en un fitxer ocult
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RodrigoBonferroni/taed2-PedestrianDetection.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "claudiamur"
os.environ["MLFLOW_TRACKING_PASSWORD"] = password
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


# Define the data loaders
batch_size_train = 2

data_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_val = torch.utils.data.DataLoader(
    validation_dataset, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=False, num_workers=2,
    collate_fn=utils.collate_fn)


tracker = EmissionsTracker(
    output_file='emissions.csv',
    on_csv_write='append',
)
tracker.start()

# Import pre-trained model mask rcnn resnet 50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device}.")


num_classes = 2
# load an instance segmentation model pre-trained on COCO
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)

# get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# now get the number of input features for the mask classifier
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
# and replace the mask predictor with a new one
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                   hidden_layer,
                                                   num_classes)

model.to(device)


# Fine tunning experiments
mlflow.set_experiment("Fine tunning")
tracker.start()
train_mlflow(model2, data_loader, data_loader_val, num_epochs=3, hidden_layer=hidden_layer,
             batch_size_train=batch_size_train, name="baseline", device=device)
tracker.stop()

torch.save(model2, 'baseline.pth')


tracker.start()
evaluation_mflow('test', data_loader_test, final_model, device)

"""
Module name: modelling_pipeline.py

This module contains the code for retraining the model using the training dataset.
"""
import configparser
import os
import sys

from codecarbon import EmissionsTracker
import dvc.api
import mlflow
import mlflow.pytorch
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from modelling import train_mlflow
from data import load_data

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
sys.path.insert(1, os.path.join(root_dir, 'src/vision'))
sys.path.insert(1, os.path.join(root_dir, 'src/data'))

from pedestrian_dataset_class import PedestrianDataset
from engine import train_one_epoch, evaluate
import utils


def main():
    # Parameters (in params.yaml)
    params = dvc.api.params_show()
    params = params["modelling"]

    batch_size_train = params["batch_size_train"]
    hidden_layer = params["hidden_layer"]  # for mask predictor
    training_epochs = params["training_epochs"]

    # Download preprocessed data
    DATA_FOLDER = os.path.join(root_dir, 'data/processed')
    training_dataset, validation_dataset, _ = load_data(DATA_FOLDER)

    # Set environment variables
    os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/RodrigoBonferroni/" \
                                        "taed2-PedestrianDetection.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = "claudiamur"
    config = configparser.ConfigParser()
    config.read(os.path.join(root_dir, 'config.ini'))
    os.environ["MLFLOW_TRACKING_PASSWORD"] = config['Credentials']['mlflow_password']
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Define the data loaders
    # Only used to train using a subset of the data (for testing purposes)
    training_dataset = torch.utils.data.Subset(training_dataset, range(0, 2))
    validation_dataset = torch.utils.data.Subset(validation_dataset, range(0, 2))
    data_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # Start tracking emissions for training
    tracker = EmissionsTracker(
        output_file=os.path.join(root_dir, 'metrics/emissions.csv'),
        on_csv_write='append',
    )
    tracker.start()

    # Import pre-trained model mask rcnn resnet 50
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    NUM_CLASSES = 2
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    # get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       NUM_CLASSES)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Fine tunning experiment
    mlflow.set_experiment("Pipeline retraining")
    tracker.start()
    train_mlflow(model, data_loader, data_loader_val, num_epochs=training_epochs,
                 hidden_layer=hidden_layer, batch_size_train=batch_size_train,
                 name="baseline", device=device, emissions_file=os.path.join(root_dir, 'metrics/emissions.csv'))
    tracker.stop()

    torch.save(model, os.path.join(root_dir, 'models/baseline_retrain.pth'))  # es sobreescriu


if __name__ == '__main__':
    main()

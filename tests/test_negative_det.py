
from PIL import Image
import pytest
import torch
import pickle
import numpy as np
import os
import torchvision.transforms as transforms

# Load the model from models directory
device = torch.device('cpu')
model = torch.load('models/baseline.pth', map_location=device)
model.eval()  # Set the model to evaluation mode


@pytest.mark.parametrize(
    "image_path",
    ["tests/img_test/negative_detection.jpeg"],
)
# Test when a pedestrian is present in the image
def test_pedestrian_detection_positive(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    # Convert the single image to a batch (size 1)
    image = [transforms.ToTensor()(image).to(device)]

    # Get model predictions for the image
    predictions = model(image)

    # Define the pedestrian class label
    pedestrian_label = 1

    # Define a confidence score threshold
    confidence_threshold = 0.8

    # Check that there are no pedestrian detections with confidence above the threshold
    no_pedestrian_detection = True
    for label, score in zip(predictions[0]['labels'], predictions[0]['scores']):
        if label == pedestrian_label and score > confidence_threshold:
            no_pedestrian_detection = False
            break  # Exit the loop if a pedestrian detection is found

    assert no_pedestrian_detection, "A pedestrian detection is present when it should be negative"

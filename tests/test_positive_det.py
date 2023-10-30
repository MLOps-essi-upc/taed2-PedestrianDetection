"""
Module Name: test_positive_det.py

Pedestrian Detection Test - Positive Detection Case

This module contains a test for the Pedestrian Detection model when a pedestrian is present in 
an image. It loads the model, processes an image known to have a pedestrian, and checks the 
model's predictions to ensure that there is at least one pedestrian detection with a confidence 
score above a specified threshold.

The test case included in this module is:
- `test_pedestrian_detection_positive`: Tests the model's ability to correctly detect a pedestrian 
when one is present in the image and checks if there is at least one true positive detection.

This module is part of the Pedestrian Detection API testing suite.

For more details on the specific test case and its assertions, refer to the function docstring below
"""

import os
from PIL import Image
import pytest
import torch
import torchvision.transforms as transforms

# Load the model from models directory
device = torch.device('cpu')
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
model_path = os.path.join(root_dir, 'models/baseline.pth')
model = torch.load(model_path, map_location=device)
model.eval()  # Set the model to evaluation mode



@pytest.mark.parametrize(
    "image_path",
    [os.path.join(root_dir, "tests/img_test/positive_detection.jpeg")],
)
# Test when a pedestrian is present in the image
def test_pedestrian_detection_positive(image_path):
    """
    Test the model's ability to correctly detect a pedestrian when one is present in the image.
    The function loads the provided image, converts it to the required format, and makes predictions 
    using the loaded model. It then checks if there is at least one pedestrian detection with a 
    confidence score above a specified threshold, ensuring that there is at least one true positive 
    detection.

    Parameters:
    - image_path (str): The path to the image known to have a pedestrian.

    Raises:
    - AssertionError: If there is no pedestrian detection or the confidence score is too low.
    """

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

    # Check if there are any pedestrian detections with confidence above the threshold
    has_pedestrian_detection = False
    for label, score in zip(predictions[0]['labels'], predictions[0]['scores']):
        if label == pedestrian_label and score >= confidence_threshold:
            has_pedestrian_detection = True
            break  # Exit the loop as soon as one positive detection is found

    assert has_pedestrian_detection, "No pedestrian detected or confidence score is too low"

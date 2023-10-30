"""
Module Name: test_negative_det.py

Pedestrian Detection Test - Positive Detection Case

This module contains a test for the Pedestrian Detection model when a pedestrian is present in 
an image. It loads the model, processes an image known to have a pedestrian, and checks the 
model's predictions toensure that there are pedestrian detections with a confidence score above 
a specified threshold.

The test cases included in this module are:
- `test_pedestrian_detection_positive`: Tests the model's ability to correctly detect a 
pedestrian when one is present in the image and checks if there are no false negative detections.

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
    [os.path.join(root_dir, "tests/img_test/negative_detection.jpeg")],
)
# Test when a pedestrian is present in the image
def test_pedestrian_detection_positive(image_path):
    """
    Test the model's ability to correctly detect a pedestrian when one is present in the image.
    The function loads the provided image, converts it to the required format, and makes predictions 
    using the loaded model. It then checks if there are no pedestrian detections with a confidence 
    score above a specified threshold, ensuring that there are no false negative detections.

    Parameters:
    - image_path (str): The path to the image known to have a pedestrian.

    Raises:
    - AssertionError: If a pedestrian detection is present when it should be negative.
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

    # Check that there are no pedestrian detections with confidence above the threshold
    no_pedestrian_detection = True
    for label, score in zip(predictions[0]['labels'], predictions[0]['scores']):
        if label == pedestrian_label and score > confidence_threshold:
            no_pedestrian_detection = False
            break  # Exit the loop if a pedestrian detection is found

    assert no_pedestrian_detection, "A pedestrian detection is present when it should be negative"

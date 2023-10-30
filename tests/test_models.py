"""
Module Name: test_models.py

Pedestrian Detection Model Performance Test

This module contains a test for evaluating the performance of the Pedestrian Detection model 
on various test images. It loads the model, processes test images, and checks the model's predictions 
against expected target results. The test assesses the model's performance by comparing predicted 
bounding boxes and masks with the expected target values.

The test cases included in this module are parameterized for different test images, each with a 
corresponding target result.

This module is part of the Pedestrian Detection API testing suite.

For more details on the specific test cases and their assertions, refer to the function docstrings below
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


# Images to test
@pytest.mark.parametrize(
    "image_path, target_path",
    [(os.path.join(root_dir, 'tests/img_test/lying.jpeg'), os.path.join(root_dir, 'tests/img_test/lying_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/night.jpeg'), os.path.join(root_dir, 'tests/img_test/night_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/sun_glare.jpeg'), os.path.join(root_dir, 'tests/img_test/sun_glare_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/weelchair.jpeg'), os.path.join(root_dir, 'tests/img_test/weelchair_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/kid.png'), os.path.join(root_dir, 'tests/img_test/kid_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/african.jpeg'), os.path.join(root_dir, 'tests/img_test/african_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/bicycle.jpeg'), os.path.join(root_dir, 'tests/img_test/bicycle_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/bicycle_tunnel.jpeg'), os.path.join(root_dir, 'tests/img_test/bicycle_tunnel_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/lot_of_people.jpeg'), os.path.join(root_dir, 'tests/img_test/lot_of_people_output.pth')),
     (os.path.join(root_dir, 'tests/img_test/bluring.jpeg'), os.path.join(root_dir, 'tests/img_test/bluring_output.pth'))]
)



def test_model_performance(image_path, target_path):
    """
    Test the performance of the Pedestrian Detection model on various test images. 
    It loads the provided image and target, makes predictions using the loaded model, and checks 
    the predictions against the expected target values. The test evaluates the model's performance 
    by comparing predicted bounding boxes and masks with the expected target values.

    Parameters:
    - image_path (str): The path to the test image.
    - target_path (str): The path to the corresponding target result for the test image.

    Raises:
    - AssertionError: If the model's predictions do not match the expected target values.
    """

    # Load the image and target
    image = Image.open(image_path).convert("RGB")
    # Convert the single image to a batch (size 1)
    image = [transforms.ToTensor()(image).to(device)]
    target = torch.load(target_path, map_location=device)

    # Get model predictions for the input image
    predictions = model(image)[0]


    # Check if the predictions are empty or not --> nothing to compare
    if not predictions['boxes'].shape[0] and not target['boxes'].shape[0]:
        return

    # Filter out predictions with scores lower than the threshold
    score_threshold = 0.8
    above_threshold = predictions['scores'] >= score_threshold
    predictions['boxes'] = predictions['boxes'][above_threshold]
    predictions['masks'] = predictions['masks'][above_threshold]

    # Check the number of predictions
    assert len(predictions['scores']) == len(
        target['scores']), "Number of predictions does not match"

    # Define a tolerance level for numerical comparisons
    bbox_tolerance = 1.0
    mask_tolerance = 0.005

    for pred_box, pred_mask, target_box, target_mask in zip(predictions['boxes'], predictions['masks'], target['boxes'], target['masks']):
        assert torch.allclose(pred_box, target_box, rtol=bbox_tolerance,
                            atol=bbox_tolerance), "Bounding boxes do not match"
        assert comparison(pred_mask, target_mask, mask_tolerance, 80), "Masks do not match"





# Compare two tensors and check if a specified percentage of elements are within a tolerance threshold
def comparison(a, b, tol, threshold_percentage):
    """
    Compare two tensors and check if a specified percentage of elements are within a tolerance 
    threshold.

    Parameters:
    - a (torch.Tensor): The first tensor for comparison.
    - b (torch.Tensor): The second tensor for comparison.
    - tol (float): The tolerance threshold for comparing elements.
    - threshold_percentage (int): The minimum percentage of elements within the tolerance threshold 
    for the comparison to be considered a match.

    Returns:
    - bool: True if the specified percentage of elements are within the tolerance 
    threshold; False otherwise.
    """

    # Calculate absolute differences between elements
    absolute_diff = torch.abs(a - b)

    # Check if the absolute differences meet the tolerance conditions
    within_tolerance = absolute_diff <= tol

    # Calculate the total number of elements in the tensor
    total_elements = a.numel()

    # Calculate the number of elements within tolerance
    elements_within_tolerance = torch.sum(within_tolerance)

    # Calculate the percentage of elements below the tolerance
    percentage = (elements_within_tolerance / total_elements) * 100.0

    return percentage.item() >= threshold_percentage  # Return the percentage as a float

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

# Images to test


@pytest.mark.parametrize(
    "image_path, target_path",
    [('tests/img_test/lying.jpeg', 'tests/img_test/lying_output.pth'),
     ('tests/img_test/night.jpeg', 'tests/img_test/night_output.pth'),
     ('tests/img_test/sun_glare.jpeg', 'tests/img_test/sun_glare_output.pth'),
     ('tests/img_test/weelchair.jpeg', 'tests/img_test/weelchair_output.pth'),
     ('tests/img_test/kid.png', 'tests/img_test/kid_output.pth'),
     ('tests/img_test/african.jpeg', 'tests/img_test/african_output.pth'),
     ('tests/img_test/bicycle.jpeg', 'tests/img_test/bicycle_output.pth'),
     ('tests/img_test/bicycle_tunnel.jpeg', 'tests/img_test/bicycle_tunnel_output.pth'),
     ('tests/img_test/lot_of_people.jpeg', 'tests/img_test/lot_of_people_output.pth'),
     ('tests/img_test/bluring.jpeg', 'tests/img_test/bluring_output.pth')]
)
def test_model_performance(image_path, target_path):
    # Load the image and target
    image = Image.open(image_path).convert("RGB")
    # Convert the single image to a batch (size 1)
    image = [transforms.ToTensor()(image).to(device)]
    target = torch.load(target_path, map_location=device)

    # Get model predictions for the input image
    predictions = model(image)

    # Define a tolerance level for numerical comparisons
    score_tolerance = 0.05
    bbox_tolerance = 1.0
    mask_tolerance = 0.05

    # Check if the predictions are empty or not --> nothing to compare
    if not predictions[0]['boxes'].shape[0] and not target['boxes'].shape[0]:
        return

    # Filter out predictions with scores lower than the threshold
    score_threshold = 0.8
    above_threshold = predictions[0]['scores'] >= score_threshold
    predictions[0]['boxes'] = predictions[0]['boxes'][above_threshold]
    predictions[0]['masks'] = predictions[0]['masks'][above_threshold]

    # Check the number of predictions
    assert len(predictions[0]['scores']) == len(
        target['scores']), "Number of predictions does not match"

    # # Compare common labels and score differences
    # common_labels = set(predictions[0]['labels']).intersection(target['labels'])
    # for label in common_labels:
    #     pred_indices = [i for i, l in enumerate(predictions[0]['labels']) if l == label]
    #     target_indices = [i for i, l in enumerate(target['labels']) if l == label]
    #     assert all(torch.isclose(predictions[0]['scores'][pred_i], target['scores'][target_i], rtol=score_tolerance, atol=score_tolerance)
    #                for pred_i, target_i in zip(pred_indices, target_indices)), "Score differences for common labels"


for pred_box, pred_mask, target_box, target_mask in zip(predictions[0]['boxes'], predictions[0]['masks'], target['boxes'], target['masks']):
    assert torch.allclose(pred_box, target_box, rtol=bbox_tolerance,
                          atol=bbox_tolerance), "Bounding boxes do not match"
    assert torch.allclose(pred_mask, target_mask, rtol=mask_tolerance,
                          atol=mask_tolerance), "Masks do not match"

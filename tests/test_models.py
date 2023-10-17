import numpy as np
from PIL import Image
from pedestrian_detector import detect_pedestrian
from your_model_module import YourPedestrianDetectionModel
from PedestrianDatasetClass import PedestrianDataset
import pytest
import torch

"""
Generar un nou notebook o fitxer, carregar el model i fer els testos(Pytest).

TESTOS:

Precisió → si està per sota del 70 % error

Mirar bona prediccio?
Si hi ha oclusions

Canvis de llum(fosc, clar)

Postura dels vianants(terra, cadira de rodes, nens)

Canvi de raça
"""

# Load the model from models directory


@pytest.fixture
def load_model():
    return torch.load('src/models/baseline.pth')


# test for multiple inputs !! --> he de buscar les fotos
# segurament els hi hagi de fer algo a les fotos pq tinguin l'estructura que cal
@pytest.mark.parametrize(
    "input, expected",
    [
        ([[6.4, 2.8, 5.6, 2.1]], 2),
        ([[5.0, 2.3, 3.3, 1.0]], 1),
        ([[4.9, 2.5, 4.5, 1.7]], 2),
    ],
)
"""
predictions = {
    'boxes': torch.tensor([[x1, y1, x2, y2], [x1, y1, x2, y2], ...]),  # Bounding boxes
    'labels': torch.tensor([label1, label2, ...]),  # Class labels
    'scores': torch.tensor([score1, score2, ...]),  # Confidence scores
    'masks': torch.tensor([mask1, mask2, ...]),  # Pixel-wise masks
}
"""


def correct_input(Image):
    # Load the image
    image = Image.open(image_path)

    # Resize the image to the model's expected size
    image = image.resize((800, 800))

    # Convert to RGB format
    image = image.convert("RGB")

    # Normalize pixel values (assuming mean and std values)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = np.array(image) / 255.0  # Scale to [0, 1]
    image = (image - mean) / std  # Normalize

    # Convert to a NumPy array
    image = np.expand_dims(image, axis=0)  # Create a batch of one

    # Now 'image' is ready for input to the Mask R-CNN ResNet-50 model
    return image

# Check the prediction is correct


def test_model_performance(model, image_path, target):
    # Load the image
    image = correct_input(image_path)

    # Get model predictions for the input image
    predictions = model(image)

    # Define a tolerance level for numerical comparisons (e.g., for bounding box coordinates)
    tolerance = 1e-5

    # Compare individual components of the predictions to the target
    assert torch.allclose(predictions['boxes'], target['boxes'], rtol=tolerance, atol=tolerance)
    assert torch.equal(predictions['labels'], target['labels'])
    assert torch.allclose(predictions['scores'], target['scores'], rtol=tolerance, atol=tolerance)
    assert torch.equal(predictions['masks'], target['masks'])


# Test when a pedestrian is present in the image
def test_pedestrian_detection_positive(model, image_path):
    # Load the image
    image = correct_input(image_path)

    # Get model predictions for the image
    predictions = model(image)

    # Define the pedestrian class label (adjust based on your specific dataset)
    pedestrian_label = 1  # Example: 1 for pedestrians

    # Define a confidence score threshold (adjust as needed)
    confidence_threshold = 0.8  # Example: 0.7

    # Check if there are any pedestrian detections with confidence above the threshold
    has_pedestrian_detection = False
    for label, score in zip(predictions['labels'], predictions['scores']):
        if label == pedestrian_label and score > confidence_threshold:
            has_pedestrian_detection = True
            break  # Exit the loop as soon as one positive detection is found

    assert has_pedestrian_detection, "No pedestrian detected or confidence score is too low"


# Test when no pedestrian is present in the image
def test_pedestrian_detection_negative(model, image_path):
    # Load the image without a pedestrian
    image = correct_input(image_path)

    # Get model predictions for the image
    predictions = model(image)

    # Define the pedestrian class label (adjust based on your specific dataset)
    pedestrian_label = 1  # Example: 1 for pedestrians

    # Define a confidence score threshold (adjust as needed)
    confidence_threshold = 0.7  # Example: 0.7

    # Check that there are no pedestrian detections with confidence above the threshold
    no_pedestrian_detection = True
    for label, score in zip(predictions['labels'], predictions['scores']):
        if label == pedestrian_label and score > confidence_threshold:
            no_pedestrian_detection = False
            break  # Exit the loop if a pedestrian detection is found

    assert no_pedestrian_detection, "A pedestrian detection is present when it should be negative"


# Test model precision --> NO ESTIC SEGURA
def calculate_precision(predictions, ground_truth):
    # Define true positives (TP) and false positives (FP)
    TP = 0
    FP = 0

    # Define a confidence score threshold for detections
    confidence_threshold = 0.7

    for prediction, annotation in zip(predictions, ground_truth):
        # Compare the class labels (assuming 1 for pedestrians)
        if prediction['label'] == 1:
            # Check if the prediction is above the confidence threshold
            if prediction['score'] >= confidence_threshold:
                # Check if there is a corresponding pedestrian in the ground truth
                if annotation['label'] == 1:
                    TP += 1  # True Positive
                else:
                    FP += 1  # False Positive

    # Calculate precision
    if TP + FP == 0:
        precision = 1.0  # If there are no predictions
    else:
        precision = TP / (TP + FP)

    return precision

# cOM PASSO DATASET??


def test_precision(model, dataset):
    # Create lists to store model predictions and ground truth annotations
    predictions = []
    ground_truth = []

    # Process each image in the dataset
    for image, annotation in dataset:
        model_output = model(image)
        predictions.append(model_output)
        ground_truth.append(annotation)

    # Calculate precision
    precision = calculate_precision(predictions, ground_truth)

    # Check if precision is greater than or equal to a threshold (e.g., 0.7)
    assert precision >= 0.7, f"Model precision is below the threshold: {precision}"


# To run the tests
# PYTHONPATH=. python3 -m pytest

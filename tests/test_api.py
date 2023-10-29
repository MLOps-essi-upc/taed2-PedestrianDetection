import imagehash
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import sys
import os
import imagehash
import requests

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(1, os.path.join(root_dir, 'src/app'))
from api import app, _load_model

# Manually load the model before running the tests
# because the client will not do the api startup
_load_model()

# Initialize the test client
client = TestClient(app)

# Define test image path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
test_image_path = os.path.join(root_dir, "tests/img_test/african.jpeg")

# Define expected image paths
expected_bb_image_path = os.path.join(root_dir, "tests/img_test/bb.png")
expected_mask_image_path = os.path.join(root_dir, "tests/img_test/mask.png")


# Test the root endpoint
def test_welcome():
    response = client.get("/")
    assert response.status_code == 200
    json = response.json()
    assert json["message"] == "OK"
    assert json["method"] == "GET"
    assert json["status-code"] == 200
    assert json["timestamp"] is not None
    assert json["url"] == "http://testserver/" 
    assert (json["data"]["welcome_message"] ==
            "Welcome to the Pedestrian Detection API! Please read the `/docs` for more information.")

# Test the /bb endpoint
def test_return_bb():
    with open(test_image_path, "rb") as image_file:
        files = {"image": ("african.jpeg", image_file)}
        response = client.post("/bb", files=files)
        assert response.status_code == 200
        json = response.json()
        assert json["message"] == "OK"
        assert json["method"] == "POST"
        assert json["status-code"] == 200
        assert json["timestamp"] is not None
        assert json["url"] == "http://testserver/bb"
        data = json["data"]
        assert "boxes" in data
        assert "scores" in data
        # Additional assertions for the specific values within the "data" dictionary
        expected_boxes = [
            [2874.278564453125, 709.43896484375, 3576.279296875, 2858.65234375]
        ]
        expected_scores = [0.9965097308158875]
        assert data["boxes"] == expected_boxes
        assert data["scores"] == expected_scores

# Test the /draw_bb endpoint
def test_draw_bb():
    with open(test_image_path, "rb") as image_file:
        files = {"image": ("african.jpeg", image_file)}
        response = client.post("/bb_draw", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

        # Load the generated image
        generated_image = Image.open(io.BytesIO(response.content))

        # Load the expected image
        expected_image = Image.open(expected_bb_image_path)

        # Calculate the perceptual hash for the generated and expected images
        hash_generated = imagehash.phash(generated_image)
        hash_expected = imagehash.phash(expected_image)
        # Compare the generated and expected images 
        # structural comparison rather than pixel-by-pixel because the bb have different colors
        assert hash_generated == hash_expected

# Test the /masks endpoint
def test_draw_masks():
    with open(test_image_path, "rb") as image_file:
        files = {"image": ("african.jpeg", image_file)}
        response = client.post("/masks", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

        # Load the generated image
        generated_image = Image.open(io.BytesIO(response.content))

        # Load the expected image
        expected_image = Image.open(expected_mask_image_path)

        # Compare the generated and expected images
        assert np.array_equal(np.array(generated_image),
                              np.array(expected_image))


# Define a function to validate error responses
def validate_error_response(response, expected_error_message):
    assert response.status_code == 400
    response_json = response.json()
    assert 'detail' in response_json
    assert expected_error_message in response_json['detail']


# Test invalid images
def test_invalid_images():
    # Create an invalid image (e.g., a text file) and send it in the request
    with open("invalid_image.txt", "w") as text_file:
        text_file.write("This is not an image.")

    # Test /bb_draw endpoint with an invalid image
    with open("invalid_image.txt", "rb") as image_file:
        files = {"image": ("invalid_image.txt", image_file)}
        response = client.post("/bb_draw", files=files)
    validate_error_response(
        response, "The input file must be an image (jpg, jpeg, png, gif)")

    # Test /bb endpoint with an invalid image
    with open("invalid_image.txt", "rb") as image_file:
        files = {"image": ("invalid_image.txt", image_file)}
        response = client.post("/bb", files=files)
    validate_error_response(
        response, "The input file must be an image (jpg, jpeg, png, gif)")

    # Test /masks endpoint with an invalid image
    with open("invalid_image.txt", "rb") as image_file:
        files = {"image": ("invalid_image.txt", image_file)}
        response = client.post("/masks", files=files)
    validate_error_response(
        response, "The input file must be an image (jpg, jpeg, png, gif)")

    # Clean up the temporary invalid image file
    try:
        os.remove("invalid_image.txt")
    except FileNotFoundError:
        pass


# Test invalid score_thres values
def test_invalid_score_thres():
    with open(test_image_path, "rb") as image_file:
        files = {"image": ("african.jpeg", image_file)}
        response = client.post("/bb_draw", files=files)

        # Test /bb_draw endpoint with an invalid score_thres
        response = client.post("/bb_draw", files=files,
                               data={"score_thres": 1.5})
        try:
            response.raise_for_status()
        except HTTPError as e:
            validate_error_response(
                e.response, "score_thres must be a number between 0 and 1")

        # Test /bb endpoint with an invalid score_thres
        response = client.post("/bb", files=files,
                               data={"score_thres": 1.5})
        try:
            response.raise_for_status()
        except HTTPError as e:
            validate_error_response(
                e.response, "score_thres must be a number between 0 and 1")

        # Test /masks endpoint with an invalid score_thres
        response = client.post("/masks", files=files,
                               data={"score_thres": 1.5})
        try:
            response.raise_for_status()
        except HTTPError as e:
            validate_error_response(
                e.response, "score_thres must be a number between 0 and 1")




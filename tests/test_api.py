from src.app.api import app  # Import your FastAPI app
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np
import sys
import os
from src.app.api import app


# Initialize the test client
client = TestClient(app)

# Define test image path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
test_image_path = os.path.join(root_dir, "tests/img_test/african.jpeg")

# Define expected image paths
expected_bb_image_path = os.path.join(root_dir, "tests/img_test/bb.png")
expected_mask_image_path = os.path.join(root_dir, "tests/img_test/mask.png")

# Define test cases for each endpoint


def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "OK"
    assert "Welcome to the Pedestrian Detection API!" in response.json()[
        "data"]["message"]


def test_return_bb():
    with open(test_image_path, "rb") as image_file:
        files = {"image": ("african.jpeg", image_file)}
        response = client.post("/bb", files=files)
        json = response.json()
        assert json["message"] == "Request failed"
        assert json["method"] == "POST"
        assert json["status-code"] == 200
        assert "url" in json
        assert "/bb?score_thres=0.8" in json["url"]
        assert json["timestamp"] is not None
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

        # Compare the generated and expected images
        assert np.array_equal(np.array(generated_image),
                              np.array(expected_image))


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


def test_error_case():
    # Create an invalid image (e.g., a text file) and send it in the request
    with open("invalid_image.txt", "w") as text_file:
        text_file.write("This is not an image.")

    with open("invalid_image.txt", "rb") as image_file:
        files = {"image": ("invalid_image.txt", image_file)}
        response = client.post("/bb_draw", files=files)

    # Ensure the response indicates an error
    json = response.json()
    assert "error" in json
    assert "Request failed" in json["message"]
    assert "method" in json
    assert json["method"] == "POST"
    assert "status-code" in json
    assert json["status-code"] == 500  # Adjust the expected status code
    assert "timestamp" in json
    assert "url" in json

    # Check for the relative path
    assert "/bb?score_thres=0.8" in json["url"]

    # Clean up the temporary invalid image file
    try:
        os.remove("invalid_image.txt")
    except FileNotFoundError:
        pass


# No need to clean up the test image file in this case
if __name__ == "__main__":
    pytest.main()

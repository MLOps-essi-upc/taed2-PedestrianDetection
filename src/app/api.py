"""Main script: it includes our API initialization and endpoints."""

from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse
from PIL import Image
from torchvision import transforms
import io
import torch
from http import HTTPStatus
from functools import wraps
from datetime import datetime
from draw_segmentation_map import draw_segmentation_map
from draw_mask import draw_mask
import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))
model_path = os.path.join(root_dir, "models/baseline.pth")

# Define application
app = FastAPI(
    title="Pedestrian detection API",
    description="This API detects and locates pedestrians within images. It accepts image inputs and returns bounding box coordinates, masks, and confidence scores for detected pedestrians.",
    version="0.1",
)

"""Function to detect pedestrians in an image"""

def detect_pedestrians(img, score_thres: float):

    # Make predictions for the image
    with torch.no_grad():
        output = app.state.model([img])[0]  # expects a list of RGB imgs and returns a list

    if output['labels'].size() == torch.Size([0]):
        return output

    # Filter predictions based on score and label thresholds
    indices_to_keep = torch.nonzero(
        torch.logical_and(
            output['labels'] == 1,
            output['scores'] >= score_thres
        )
    ).squeeze()

    # when there is only one index: tensor(1) --> tensor([1])
    if indices_to_keep.dim() == 0:
        indices_to_keep = torch.tensor([indices_to_keep.item()])

    # Remove elements that don't meet the conditions from each tensor
    for key in ['scores', 'labels', 'boxes', 'masks']:
        output[key] = output[key][indices_to_keep]  # mantain list structure

    return output


"""Decorator to construct a JSON response for an endpoint's results"""


def construct_response(f):
    @wraps(f)
    def wrap(request: Request, *args, **kwargs):

        results = f(request, *args, **kwargs)

        # Construct response for a successful request
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data if available
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrap


"""Load the model on startup"""


@app.on_event("startup")
def _load_model():
    # Load the model
    device = torch.device('cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()  # Set the model to evaluation mode
    app.state.model = model


"""Root endpoint with a welcome message"""


@app.get("/", tags=["General"])
@construct_response
def _index(request: Request):
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to the Pedestrian Detection API! Please read the `/docs` for more information."},
    }

    return response


"""Detect pedestrians with bounding boxes and return an image endpoint"""


@app.post("/bb_draw", tags=["bb"])
def draw_bb(request: Request, image: UploadFile, score_thres: float = 0.8):
    try:
        # Read and preprocess the uploaded image
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        img = transforms.ToTensor()(img)

        # Perform pedestrian detection with bounding boxes
        result = detect_pedestrians(img, score_thres)

        # Draw bounding boxes on the image using the provided function
        img_with_bb = draw_segmentation_map(img, result)

        # Convert the NumPy array to a PIL Image
        img_with_bb_pil = Image.fromarray((img_with_bb * 255).astype('uint8'))

        # Save the image to a temporary file
        image_path = "image.png"  # Choose an appropriate path and format

        # Save the PIL image to the specified path
        img_with_bb_pil.save(image_path, "PNG")

        # Open the saved image file
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Return the image as a file for download using StreamingResponse
        image_response = StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

        return image_response
    except Exception as e:
        # Construct an error response
        error_response = {
            "message": "Request failed",
            "method": request.method,
            "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
            "data": {"error": str(e)}
        }
        return error_response


"""Detect pedestrians with bounding boxes and return coordinates and scores endpoint"""


@app.post("/bb", tags=["bb"])
@construct_response
def return_bb(request: Request, image: UploadFile, score_thres: float = 0.8):
    try:
        # Read and preprocess the uploaded image
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        img = transforms.ToTensor()(img)

        # Perform pedestrian detection with bounding boxes
        result = detect_pedestrians(img, score_thres)

        # Construct a successful response
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {
                "boxes": result['boxes'].tolist(),
                "scores": result['scores'].tolist(),
            }
        }

        return response
    except Exception as e:
        # Construct an error response
        error_response = {
            "message": "Request failed",
            "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "data": {"error": str(e)}
        }
        return error_response


"""Detect pedestrians with masks and return image endpoint"""


@app.post("/masks", tags=["mask"])
def draw_mask(request: Request, image: UploadFile, score_thres: float = 0.8):
    try:
        # Read and preprocess the uploaded image
        img = Image.open(io.BytesIO(image.file.read())).convert("RGB")
        img = transforms.ToTensor()(img)

        # Perform pedestrian detection with bounding boxes
        result = detect_pedestrians(img, score_thres)

        # Draw bounding boxes on the image using the provided function
        img_with_bb = draw_mask(img, result)

        # Convert the NumPy array to a PIL Image
        img_with_bb_pil = Image.fromarray((img_with_bb * 255).astype('uint8'))

        # Save the image to a temporary file
        image_path = "image.png"  # Choose an appropriate path and format

        # Save the PIL image to the specified path
        img_with_bb_pil.save(image_path, "PNG")

        # Open the saved image file
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        # Return the image as a file for download using StreamingResponse
        image_response = StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")

        return image_response
    except Exception as e:
        # Construct an error response
        error_response = {
            "message": "Request failed",
            "method": request.method,
            "status-code": HTTPStatus.INTERNAL_SERVER_ERROR,
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
            "data": {"error": str(e)}
        }
        return error_response

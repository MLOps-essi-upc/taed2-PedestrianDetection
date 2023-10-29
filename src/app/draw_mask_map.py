"""
Module Name: draw_mask_map.py

This module provides utility functions for working with COCO datasets
and segmentation map visualization drawning a mask.
"""

import cv2
import numpy as np
import torch


COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
              'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COLORS = np.random.uniform(0, 255, size=(len(COCO_NAMES), 3)).astype(int)


def draw_mask_map(image, target, score_thres=0.8):
    """
    This function takes an input image and a target (typically from a COCO dataset) and 
    draws masks on the image for segmentation visualization.

    Parameters:
        - image (torch.Tensor): The input image as a PyTorch tensor.
        - target (dict): A dictionary containing information about the detected objects, 
        including masks, bounding boxes, labels, and scores.
        - score_thres (float): A threshold for object detection scores. Objects with scores 
        below this threshold will not be drawn on the image.

    Returns:
        - np.ndarray: The image with masks drawn, represented as a NumPy array with 
        values in the range [0, 1].
    """

    # Convert back to numpy arrays
    _image = np.copy(image.cpu().detach().numpy().transpose(1, 2, 0)*255)

    if target['labels'].size() == torch.Size([0]):  # no prediction --> all background
        _image = 0.3*_image
    else:
        _masks = np.copy(target['masks'].cpu().detach().numpy().astype(np.float32))
        _boxes = np.copy(target['boxes'].cpu().detach().numpy().astype(int))
        _labels = np.copy(target['labels'].cpu().detach().numpy().astype(int))
        if "scores" in target:
            _scores = np.copy(target["scores"].cpu().detach().numpy())
        else:
            _scores = np.ones(len(_masks), dtype=np.float32)

        # Add mask if _scores
        m = np.zeros_like(_masks[0].squeeze())
        for i, mask in enumerate(_masks):
            if _scores[i] > score_thres:
                m = m + mask

        # Make sure m is the right shape
        m = m.squeeze()

        # dark pixel outside masks
        _image[m < 0.5] = 0.3*_image[m < 0.5]

        # convert from RGB to OpenCV BGR and back (cv2.rectangle is just too picky)
        # pylint: disable=no-member
        _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
        _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)
        # pylint: enable=no-member  # Re-enable pylint checks

    return _image/255

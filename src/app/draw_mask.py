"""
Module Name: coco_utils.py

This module provides utility functions for working with COCO datasets
and segmentation map visualization.
"""

import random
import cv2
import numpy as np


COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
              'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
              'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
              'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
              'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
              'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
              'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
              'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
              'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
              'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
              'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COLORS = np.random.uniform(0, 255, size=(len(COCO_NAMES), 3)).astype(int)


def draw_mask(image, target, score_thres=0.8):

    # Convert back to numpy arrays
    _image = np.copy(image.cpu().detach().numpy().transpose(1, 2, 0)*255)
    _masks = np.copy(target['masks'].cpu().detach().numpy().astype(np.float32))
    _boxes = np.copy(target['boxes'].cpu().detach().numpy().astype(int))
    _labels = np.copy(target['labels'].cpu().detach().numpy().astype(int))
    if "scores" in target:
        _scores = np.copy(target["scores"].cpu().detach().numpy())
    else:
        _scores = np.ones(len(_masks), dtype=np.float32)

    alpha = 0.3

    label_names = [COCO_NAMES[i] for i in _labels]

    # Add mask if _scores
    m = np.zeros_like(_masks[0].squeeze())
    for i in range(len(_masks)):
        if _scores[i] > score_thres:
            m = m + _masks[i]

    # Make sure m is the right shape
    m = m.squeeze()

    # dark pixel outside masks
    _image[m < 0.5] = 0.3*_image[m < 0.5]

    # convert from RGB to OpenCV BGR and back (cv2.rectangle is just too picky)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

    for i in range(len(_masks)):
        if _scores[i] > score_thres:
            # apply a randon color to each object
            color = COLORS[random.randrange(0, len(COLORS))].tolist()

            # draw the bounding boxes around the objects
            # cv2.rectangle(_image, _boxes[i][0:2], _boxes[i][2:4], color=color, thickness=2)
            # put the label text above the objects
            cv2.putText(_image, label_names[i], (_boxes[i][0], _boxes[i][1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                        thickness=1, lineType=cv2.LINE_AA)

    return _image/255

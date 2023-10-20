"""
Module Name: pedestrian_dataset_class.py

This module defines the PedestrianDataset class, which is used to create a custom 
dataset for pedestrian detection tasks.
"""

import os
import random
import numpy as np

import torch
import torch.utils.data

from PIL import Image

random.seed(356)

class PedestrianDataset(torch.utils.data.Dataset):
    """
    A custom PyTorch dataset class for pedestrian detection.

    Args:
    root (str): The root directory containing the dataset.
    transforms (callable, optional): A function/transform to apply to the data.

    Attributes:
    root (str): The root directory of the dataset.
    transforms (callable, optional): The data transformation function.
    imgs (list): A list of image file names.
    masks (list): A list of mask file names.

    Methods:
    __getitem__(self, idx): Retrieve an item from the dataset.
    __len__(self): Get the length of the dataset.

    This dataset loads images and their corresponding masks and annotations, and provides them 
    as input data for pedestrian detection tasks.
    """

    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # The images are loaded and sorted to make sure they match with the appropiate mask
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # The images and masks are loaded
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # The mask is not converted to RGB given that each color corresponds to an object,
        # the background being 0
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # The objects are coded with different colors
        obj_ids = np.unique(mask)
        # The first id is not necessary as it is the background
        obj_ids = obj_ids[1:]

        # The color coded mask is separated into a binary mask
        masks = mask == obj_ids[:, None, None]

        # We get each mask's bounding box coordinates
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # There is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # We assume that all instances are not crowded
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
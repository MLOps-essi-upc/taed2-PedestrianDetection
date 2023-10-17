import cv2
import dvc.api
import os
import numpy as np
import PedestrianDatasetClass
import pickle
from PIL import Image
import random
import sys
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset
import transforms as T

def get_transform(train, transform_value = 1):
    transforms = []
    # Transforms that are applied to all images from the folder (convert to tensor and floats)
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))

    # Transforms to augment the raw data
    if train:
      if transform_value == 1:
        transforms.append(T.RandomHorizontalFlip(p=1))

      elif transform_value == 2:
        transforms.append(T.RandomShortestSize(120,800))

      else:
        transforms.append(T.RandomPhotometricDistort(p = 1))


    return T.Compose(transforms)


def draw_segmentation_map(image, target, score_thres=0.8):


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

    # It's converted to arrays
    _image = np.copy(image.cpu().detach().numpy().transpose(1,2,0)*255)
    _masks = np.copy(target['masks'].cpu().detach().numpy().astype(np.float32))
    _boxes = np.copy(target['boxes'].cpu().detach().numpy().astype(int))
    _labels = np.copy(target['labels'].cpu().detach().numpy().astype(int))
    if "scores" in target:
      _scores = np.copy(target["scores"].cpu().detach().numpy())
    else:
      _scores = np.ones(len(_masks),dtype=np.float32)

    alpha = 0.3

    label_names = [COCO_NAMES[i] for i in _labels]

    # The mask is added only if the score surpasses the threshold
    m = np.zeros_like(_masks[0].squeeze())
    for i in range(len(_masks)):
      if _scores[i] > score_thres:
        m = m + _masks[i]

    # Make sure the m has the correct size (no dimension at 1)
    m = m.squeeze()

    # The pixels outside of the image are darkened
    _image[m<0.5] = 0.3*_image[m<0.5]

    # We transform from RGB to OpenCV BGR and back
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB)

    for i in range(len(_masks)):
      if _scores[i] > score_thres:
        # Apply a random color to each object
        color = COLORS[random.randrange(0, len(COLORS))].tolist()

        # The bounding boxes are drawn around the objects
        cv2.rectangle(_image, _boxes[i][0:2], _boxes[i][2:4], color=color, thickness=2)
        # We add the class label above the objects
        cv2.putText(_image , label_names[i], (_boxes[i][0], _boxes[i][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                    thickness=1, lineType=cv2.LINE_AA)

    return _image/255








def main():
    params = dvc.api.params_show()
    seed = params["seed"]
    training_size = params["training_size"]
    validation_size = params["validation_size"]
    testing_size = params["testing_size"]

    random.seed(seed)        # Could be replaced by a params.yaml file
    DATA_FOLDER = "../data/raw/Dataset_FudanPed"
    dataset_whole = PedestrianDatasetClass.PedestrianDataset(DATA_FOLDER, get_transform(train = False, transform_value = 0))

    training_dataset0, validation_dataset, testing_dataset = torch.utils.data.dataset.random_split(dataset_whole, [training_size, validation_size, testing_size], generator = torch.Generator().manual_seed(seed))
    training_dataset1, _, _ = torch.utils.data.dataset.random_split(PedestrianDatasetClass.PedestrianDataset(DATA_FOLDER,get_transform(train = 1, transform_value = 1)), [training_size, validation_size, testing_size], generator = torch.Generator().manual_seed(seed))
    training_dataset2, _, _ = torch.utils.data.dataset.random_split(PedestrianDatasetClass.PedestrianDataset(DATA_FOLDER,get_transform(train = 1, transform_value = 2)), [training_size, validation_size, testing_size], generator = torch.Generator().manual_seed(seed))
    training_dataset3, _, _ = torch.utils.data.dataset.random_split(PedestrianDatasetClass.PedestrianDataset(DATA_FOLDER,get_transform(train = 1, transform_value = 3)), [training_size, validation_size, testing_size], generator = torch.Generator().manual_seed(seed))
    # A bigger training dataset is created by combining all four smaller training datasets.

    training_dataset = ConcatDataset([training_dataset0, training_dataset1, training_dataset2, training_dataset3])

    train_path = "../data/processed/training_datasets.pkl"
    with open(train_path, 'wb') as file:
        pickle.dump(training_dataset, file)
    validation_path = "../data/processed/validation_datasets.pkl"
    with open(validation_path, 'wb') as file:
        pickle.dump(validation_dataset, file)
    testing_path = "../data/processed/testing_datasets.pkl"
    with open(testing_path, 'wb') as file:
        pickle.dump(testing_dataset, file)


if __name__ == "__main__":
    main()
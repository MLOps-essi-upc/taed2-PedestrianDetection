"""
Module Name: data_preprocessing.py

This module defines the code used to preprocess the data used in the project.
The data will be loaded, augmented and saved in the established folders.
"""
import pickle
import os
import random
import sys

import dvc.api
import torch
import torch.utils.data
from torch.utils.data import ConcatDataset

sys.path.insert(1, os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')), 'vision'))

import transforms as T
import pedestrian_dataset_class


def get_transform(train, transform_value = 1):
    """
    This function applies the necessary transformations to the data.
    """
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

def main():
    """
    This code will load the data, augment it and save it in the established folders.
    """
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))

    # Add to root_dir the path to the processed data folder
    data_folder = os.path.join(root_dir, 'data/raw/Dataset_FudanPed')


    params = dvc.api.params_show()
    params = params["preprocessing_data"]
    seed = params["seed"]
    training_size = params["training_size"]
    validation_size = params["validation_size"]
    testing_size = params["testing_size"]

    random.seed(seed)        # Could be replaced by a params.yaml file
    dataset_whole = pedestrian_dataset_class.PedestrianDataset(data_folder,
              get_transform(train = False, transform_value = 0))
    training_dataset0, validation_dataset, testing_dataset = torch.utils.data.dataset.random_split(
                                                    dataset_whole,
                                                    [training_size, validation_size, testing_size],
                                                    generator = torch.Generator().manual_seed(seed))
    training_dataset1, _, _ = torch.utils.data.dataset.random_split(
                                                    pedestrian_dataset_class.PedestrianDataset(
                                                        data_folder,
                                                        get_transform(train = 1,
                                                                      transform_value = 1)),
                                                    [training_size, validation_size, testing_size],
                                                    generator = torch.Generator().manual_seed(seed))
    training_dataset2, _, _ = torch.utils.data.dataset.random_split(
                                                    pedestrian_dataset_class.PedestrianDataset(
                                                        data_folder,
                                                        get_transform(train = 1,
                                                                      transform_value = 2)),
                                                    [training_size, validation_size, testing_size],
                                                    generator = torch.Generator().manual_seed(seed))
    training_dataset3, _, _ = torch.utils.data.dataset.random_split(
                                                    pedestrian_dataset_class.PedestrianDataset(
                                                        data_folder,
                                                        get_transform(train = 1,
                                                                      transform_value = 3)),
                                                    [training_size, validation_size, testing_size],
                                                    generator = torch.Generator().manual_seed(seed))
    # A bigger training dataset is created by combining all four smaller training datasets.

    training_dataset = ConcatDataset(
    [training_dataset0, training_dataset1, training_dataset2, training_dataset3])

    processed_data_folder = os.path.join(root_dir, 'data/processed')

    os.makedirs(processed_data_folder, exist_ok=True)
    train_path = os.path.join(processed_data_folder, 'training_dataset.pkl')
    with open(train_path, 'wb') as file:
        pickle.dump(training_dataset, file)
    validation_path = os.path.join(processed_data_folder, 'validation_dataset.pkl')
    with open(validation_path, 'wb') as file:
        pickle.dump(validation_dataset, file)
    testing_path = os.path.join(processed_data_folder, 'testing_dataset.pkl')
    with open(testing_path, 'wb') as file:
        pickle.dump(testing_dataset, file)


if __name__ == "__main__":
    main()

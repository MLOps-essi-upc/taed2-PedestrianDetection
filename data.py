"""
Module Name: data.py

This module contains functions for downloading preprocessed 
data.
"""

import pickle

def load_data(data_folder):
    """
    Load preprocessed datasets from the specified folder.

    Args:
    data_folder (str): The path to the folder containing the preprocessed datasets.

    Returns:
    tuple: A tuple containing the training, validation, and testing datasets loaded from 
    pickle files.
    """
    # Load each dataset from their respective file.
    # Download preprocessed data

    training_file_path = data_folder + '/training_dataset.pkl'
    validation_file_path = data_folder + '/validation_dataset.pkl'
    testing_file_path = data_folder + '/testing_dataset.pkl'

    with open(training_file_path, 'rb') as file:
        training_dataset = pickle.load(file)

    with open(validation_file_path, 'rb') as file:
        validation_dataset = pickle.load(file)

    with open(testing_file_path, 'rb') as file:
        testing_dataset = pickle.load(file)

    return training_dataset, validation_dataset, testing_dataset

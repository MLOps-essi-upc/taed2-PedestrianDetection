"""
Module name: validate.py

This module defines the code used to validate the data used in the project.
The data will be validated using Great Expectations.
"""
import os
import pickle

import great_expectations as gx
import pandas as pd

# from src import ROOT_DIR
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))

# Add to root_dir the path to the processed data folder
data_dir = os.path.join(root_dir, 'data/processed')

context = gx.get_context()
context.add_or_update_expectation_suite("pennfudan_training_suite")
datasource = context.sources.add_or_update_pandas(name="pennfudan_dataset")

# Using this code we load each dataset from their respective file.
with open(os.path.join(data_dir, 'training_dataset.pkl'), 'rb') as file:
    training_dataset = pickle.load(file)
    file.close()
with open(os.path.join(data_dir, 'validation_dataset.pkl'), 'rb') as file:
    validation_dataset = pickle.load(file)
    file.close()
with open(os.path.join(data_dir, 'testing_dataset.pkl'), 'rb') as file:
    testing_dataset = pickle.load(file)
    file.close()

tensor_list = []
A_list = []
B_list = []
C_list = []
D_list = []
E_list = []
F_list = []

# Iterate through the tensor dataset and extract the values
for item in training_dataset:
    tensor_list.append(str(item[0]))
    dictionary = item[1]
    A_list.append(str(dictionary['boxes']))
    B_list.append(str(dictionary['labels']))
    C_list.append(str(dictionary['masks']))
    D_list.append(str(dictionary['image_id']))
    E_list.append(str(dictionary['area']))
    F_list.append(str(dictionary['iscrowd']))

# Create a dataframe with the extracted values
data = {
    'Tensor': tensor_list,
    'Boxes': A_list,
    'Labels': B_list,
    'Masks': C_list,
    'Image_id': D_list,
    'Area': E_list,
    'Iscrowd': F_list
}

train = pd.DataFrame(data)

# Add the dataframe to the datasource
data_asset = datasource.add_dataframe_asset(name="training", dataframe=train)

# Build the batch request
batch_request = data_asset.build_batch_request()

# Create the validator
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name="pennfudan_training_suite",
    datasource_name="pennfudan_dataset",
    data_asset_name="training"
)

# Add the expectations
validator.expect_table_columns_to_match_ordered_list(
    column_list=[
        "Tensor",
        "Boxes",
        "Labels",
        "Masks",
        "Image_id",
        "Area",
        "Iscrowd",
    ]
)

validator.expect_column_values_to_be_unique("Tensor")
validator.expect_column_values_to_not_be_null("Tensor")
validator.expect_column_values_to_not_be_null("Image_id")
validator.expect_column_values_to_not_be_null("Masks")
validator.expect_column_values_to_not_be_null("Boxes")

validator.save_expectation_suite(discard_failed_expectations=False)

# Add the checkpoint
checkpoint = context.add_or_update_checkpoint(
    name="my_checkpoint",
    validator=validator,
)

checkpoint_result = checkpoint.run()
context.view_validation_result(checkpoint_result)

import great_expectations as gx
import pandas as pd
import pickle

context = gx.get_context()
context.add_or_update_expectation_suite("pennfudan_training_suite")
datasource = context.sources.add_or_update_pandas(name="pennfudan_dataset")

# Using this code we load each dataset from their respective file.

with open('../../data/processed/training_dataset.pkl', 'rb') as file:
    training_dataset = pickle.load(file)
    file.close()
with open('../../data/processed/validation_dataset.pkl', 'rb') as file:
    validation_dataset = pickle.load(file)
    file.close()
with open('../../data/processed/testing_dataset.pkl', 'rb') as file:
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
    tensor_list.append(item[0])
    dictionary = item[1]
    A_list.append(dictionary['boxes'])
    B_list.append(dictionary['labels'])
    C_list.append(dictionary['masks'])
    D_list.append(dictionary['image_id'])
    E_list.append(dictionary['area'])
    F_list.append(dictionary['iscrowd'])

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
validator.expect_column_values_to_be_unique("Image_id")
validator.expect_column_values_to_not_be_null("Image_id")

validator.save_expectation_suite(discard_failed_expectations=False)

# Add the checkpoint
checkpoint = context.add_or_update_checkpoint(
    name="my_checkpoint",
    validator=validator,
)

checkpoint_result = checkpoint.run()
context.view_validation_result(checkpoint_result)



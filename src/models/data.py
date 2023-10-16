import pickle
# from PedestrianDatasetClass import PedestrianDataset  # Import the required class


def load_data(DATA_FOLDER):
    # Load each dataset from their respective file.
    # Download preprocessed data
    with open(DATA_FOLDER + '/training_dataset.pkl', 'rb') as file:
        training_dataset = pickle.load(file)

    with open(DATA_FOLDER + '/validation_dataset.pkl', 'rb') as file:
        validation_dataset = pickle.load(file)

    with open(DATA_FOLDER + '/testing_dataset.pkl', 'rb') as file:
        testing_dataset = pickle.load(file)

    return training_dataset, validation_dataset, testing_dataset

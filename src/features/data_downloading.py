"""
Module Name: data_downloading.py

This module defines the code used to download the data used in the project.
The data will be downloaded from Kaggle and unzipped in the established folders.
"""
import os
import shutil
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    """
    This code will download the dataset from Kaggle and unzip it in the established folders.
    If the dataset is already in /raw, it will delete it and substitute the data.
    """

    # Download the dataset from Kaggle
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('psvishnu/pennfudan-database-for-pedestrian-detection-zip', path=".")

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'*2))


    source_zipfile = "./pennfudan-database-for-pedestrian-detection-zip.zip"
    destination_folder_pngs = os.path.join(root_dir, 'data/raw/Dataset_FudanPed/PNGImages')
    destination_folder_masks = os.path.join(root_dir, 'data/raw/Dataset_FudanPed/PedMasks')

    if os.path.exists(destination_folder_masks):
        shutil.rmtree(destination_folder_masks)
    if os.path.exists(destination_folder_pngs):
        shutil.rmtree(destination_folder_pngs)

    os.makedirs(destination_folder_pngs, exist_ok=True)
    os.makedirs(destination_folder_masks, exist_ok=True)

    with zipfile.ZipFile(source_zipfile, 'r') as source_zip:
        # Iterate through the files in the zip archive
        for file_info in source_zip.infolist():
            # Check if the file is within the specified folder and not in other folders
            if file_info.filename.startswith("PennFudanPed/PedMasks/"):
                # Build the full path to the destination file
                destination_file = os.path.join(destination_folder_masks,
                                                os.path.basename(file_info.filename))

                # Extract the file to the destination folder
                source_zip.extract(file_info, destination_folder_masks)

                os.rename(os.path.join(destination_folder_masks, file_info.filename),
                          destination_file)

            elif file_info.filename.startswith("PennFudanPed/PNGImages/"):
                # Build the full path to the destination file
                destination_file = os.path.join(destination_folder_pngs,
                                                os.path.basename(file_info.filename))
                # Extract the file to the destination folder
                source_zip.extract(file_info, destination_folder_pngs)

                os.rename(os.path.join(destination_folder_pngs, file_info.filename),
                          destination_file)
    os.remove(source_zipfile)
    shutil.rmtree(os.path.join(root_dir, 'data/raw/Dataset_FudanPed/PedMasks/PennFudanPed'))
    shutil.rmtree(os.path.join(root_dir, 'data/raw/Dataset_FudanPed/PNGImages/PennFudanPed'))

if __name__ == "__main__":
    main()

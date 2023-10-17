from kaggle.api.kaggle_api_extended import KaggleApi
import os
import shutil
import zipfile

def main():
# This code will download the dataset from Kaggle and unzip it in the established folders. If the dataset is already in /raw, it will delete it and substitute the data.

    # Download the dataset from Kaggle
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('psvishnu/pennfudan-database-for-pedestrian-detection-zip', path=".")


    source_zipfile = "./pennfudan-database-for-pedestrian-detection-zip.zip"
    destination_folder_pngs = "../data/raw/Dataset_FudanPed/PNGImages"
    destination_folder_masks = "../data/raw/Dataset_FudanPed/PedMasks"
    

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
            if file_info.filename.startswith(f"PennFudanPed/PedMasks/"):
                # Build the full path to the destination file
                destination_file = os.path.join(destination_folder_masks, os.path.basename(file_info.filename))
                
                # Extract the file to the destination folder
                source_zip.extract(file_info, destination_folder_masks)
                
                os.rename(os.path.join(destination_folder_masks, file_info.filename), destination_file)

            elif file_info.filename.startswith(f"PennFudanPed/PNGImages/"):
                # Build the full path to the destination file
                destination_file = os.path.join(destination_folder_pngs, os.path.basename(file_info.filename))
                # Extract the file to the destination folder
                source_zip.extract(file_info, destination_folder_pngs)
                
                os.rename(os.path.join(destination_folder_pngs, file_info.filename), destination_file)
    os.remove(source_zipfile)
    shutil.rmtree("../data/raw/Dataset_FudanPed/PNGImages/PennFudanPed" )
    shutil.rmtree("../data/raw/Dataset_FudanPed/PedMasks/PennFudanPed" )

    return  # Aqui se podría hacer un primer load de los datoa desde su nueva carpeta y luego lodearlos otra vez,

if __name__ == "__main__":
    main()
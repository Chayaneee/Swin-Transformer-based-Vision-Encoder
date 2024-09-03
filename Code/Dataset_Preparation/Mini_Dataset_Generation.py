import pandas as pd
import os
import shutil
from tqdm import tqdm

# Load the CSV file into a DataFrame
data = pd.read_csv("mimic-cxr-chexpert_GPTAPI_binary_label_filtered_no_overlap.csv")  # Update with the path to your CSV file
data_dir = "/mnt/data/datasets/"

# Assuming the CSV file has columns "image_path" and "label", adjust accordingly
image_paths = data["File_Path"]
labels = data["Binary_Label"]

# Define the paths for the label folders
label0_folder = "/mnt/data/chayan/Mini-MIMIC/Train/Normal"  # Update with the path where you want to store label 0 images
label1_folder = "/mnt/data/chayan/Mini-MIMIC/Train/Abnormal"  # Update with the path where you want to store label 1 images

# Create label folders if they don't exist
os.makedirs(label0_folder, exist_ok=True)
os.makedirs(label1_folder, exist_ok=True)

# Iterate through each row in the DataFrame
for image_path, label in tqdm(zip(image_paths, labels)):
    # Construct the destination folder based on the label
    destination_folder = label0_folder if label == 0 else label1_folder
    
    # Extract the image file name
    image_filename = os.path.join(data_dir, image_path)
    
    # Construct the destination path
    destination_path = destination_folder
    
    # Move the image to the destination folder
    shutil.copy(image_filename, destination_path)

print("Images moved to label folders successfully.")

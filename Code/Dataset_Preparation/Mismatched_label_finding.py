import os
import csv
import pandas as pd
import numpy as  np

image_df = pd.read_csv('/content/drive/MyDrive/MIMIC_CXR/Final_MIMIC_Dataset_Impression_GPTAPI-Labeling_20K.csv')

image_df2 = pd.read_csv('/content/drive/MyDrive/MIMIC_CXR/Final_MIMIC_Dataset_LLama2-13B-Labeling_20K.csv')

image_df3 = pd.read_csv('/content/drive/MyDrive/MIMIC_CXR/Final_MIMIC_Dataset_LLama2-7B-Labeling_20K.csv')

chexpert_df = pd.read_csv('/content/drive/MyDrive/MIMIC_CXR/mimic-cxr-chexpert_binary_label.csv')

image_df['subject_id'] = image_df['File_Path'].str.extract(r'\/p(\d+)\/')
image_df['study_id'] = image_df['File_Path'].str.extract(r'\/s(\d+)\/')
image_df['subject_id'] = image_df['subject_id'].astype(int)
image_df['study_id'] = image_df['study_id'].astype(int)

merged_df = pd.merge(image_df, chexpert_df, on=['subject_id', 'study_id'], how='left')

image_df['Binary_Label'] = merged_df['Binary_Label'].fillna(-1)

def find_non_integer_samples_in_column(df_file, column_name):
    # Read the CSV file into a DataFrame
    # df = pd.read_csv(csv_file)

    # Filter the DataFrame to include only the specified column
    column_data = df_file[column_name]

    # Find the indices of non-integer samples in the column
    non_integer_indices = column_data.index[column_data.apply(lambda x: not isinstance(x, int))].tolist()

    # Create a dictionary to store the non-integer samples along with their indices
    non_integer_samples = {index: column_data[index] for index in non_integer_indices}

    return non_integer_samples
    
non_integer_samples_in_column = find_non_integer_samples_in_column(image_df, "GT-Label")
# print("Non-integer samples in column '{}':".format(column_name))
for index, value in non_integer_samples_in_column.items():
    print("Index:", index, "Value:", value)
    
GPT_label = image_df['GT-Label'].astype(int)
CheXpert_label = image_df['Binary_Label']

mismatched_indices = []
mismatched_labels_GPT_label = []
mismatched_labels_CheXpert_label = []

# Find mismatched labels
for i in range(len(GPT_label)):
    if GPT_label[i] != CheXpert_label[i]:
        mismatched_indices.append(i)
        mismatched_labels_GPT_label.append(GPT_label[i])
        mismatched_labels_CheXpert_label.append(CheXpert_label[i])

# Print mismatched indices and corresponding labels
print("Total Mismatched labels: ", len(mismatched_indices))
# for i, index in enumerate(mismatched_indices):
#     print(f"Index: {index}, Model 1 label: {mismatched_labels_GPT_label[i]}, Model 2 label: {mismatched_labels_CheXpert_label[i]}")
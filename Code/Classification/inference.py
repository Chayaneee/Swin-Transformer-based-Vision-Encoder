import os
import torch
import transformers
from transformers import pipeline, AutoTokenizer
import pandas as pd
from tqdm import tqdm

import pydicom
import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore')


import datasets
from datasets import load_dataset

from PIL import Image


from transformers import pipeline

pipe = pipeline("image-classification", model= "/home/chayan/MIMIC-Dataset/Codes/-finetuned-mimic-filtered-Linear-v3/")

image_path = "/mnt/data/chayan/Mini-MIMIC/Test/Normal/0eeed2ac-7c2c8b80-4a7de4c0-bf204a98-3f015f56.jpg"

result= pipe(image_path)

print("Prediction with Probability Score:",result)

a = result[0]['label']

if a == 'Normal':
  label = 0
else:
  label = 1
   
print("Prediction Label ID: ",label)


data = pd.read_csv("/home/chayan/MIMIC-Dataset/Data/mimic-cxr-chexpert-test_set-binary_label.csv")

data_dir = "/mnt/data/datasets/"

image_name = data['File_Path']

print(len(image_name))

predicted_label = []
predicted_label_score = []

for image in tqdm(image_name):
    
    image_path = os.path.join(data_dir, image)
    pipe = pipeline("image-classification", model= "/home/chayan/MIMIC-Dataset/Codes/-finetuned-mimic-filtered-Linear-v3/")
    result= pipe(image_path)
    predicted_label_score.append(result)
    
    predict = result[0]['label']
    if predict == 'Normal':
      label = 0
    else:
      label = 1
    predicted_label.append(label)
    
data['Predicted_Label'] = predicted_label    
data['Predicted_Label_score'] = predicted_label_score


data.to_csv("Mimic-test-set-prediction_LRS_linear_Swin-base-384-v3.csv", index=False)    
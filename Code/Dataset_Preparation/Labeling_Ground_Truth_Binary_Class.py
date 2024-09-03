import pandas as pd
from tqdm import tqdm
import csv

from openai import OpenAI
client = OpenAI()

data = pd.read_csv("/home/chayan/MIMIC-Dataset/Data/Final_MIMIC_Dataset_Findings_Impression.csv")

Original_report = data["Findings"].reset_index(drop=True)  
file_path = data["File_Path"].reset_index(drop=True) 

print(Original_report[100])
print(len(Original_report))

original_label = []


print("Labeling Original Report ......")

for j in tqdm(range(len(Original_report))):
  completion = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=[
    {"role": "system", "content": "Please identify the following findings as normal or abnormal class. If it is abnormal class write only 1 and if it is normal write only 0. The response should be either 0 or 1. Low Lungs volume may indicate some abnormalities as well."}, # Rewrite the following radiology findings in a single paragraph maintaining the sequence of the order of lungs, heart, abdomen, and others.
    {"role": "user", "content": Original_report[j]}
   ])
  label = completion.choices[0].message.content 
  original_label.append([file_path[j], Original_report[j], label])


csv_file_path = "/home/chayan/MIMIC-Dataset/Data/Final_MIMIC_Dataset_Findings_Impression_GT-Labeling.csv"

# Write the results to a CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['File_Path', 'FIndings', 'GT-Label'])
    # Write each row of the results list to the CSV file
    for result in original_label:
        writer.writerow(result)

print("CSV file created successfully.")
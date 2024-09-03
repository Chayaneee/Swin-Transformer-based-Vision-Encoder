import os
import pandas as pd

df = pd.read_csv("/home/chayan/MIMIC-Dataset/Data/MIMIC_Dataset.csv")

print(" \n Length of Dataset before Preprocessing: ", len(df))

# Define a function to remove the sentence containing "findings were discussed with dr."
def remove_sentence(text):
    if isinstance(text, str):
        sentences = text.split('.')
        filtered_sentences = [sentence.strip() for sentence in sentences if "findings were discussed with dr" not in sentence.lower()]
        filtered_sentences = [sentence.strip() for sentence in sentences if "findings were conveyed by dr" not in sentence.lower()]
        return '. '.join(filtered_sentences)
    else:
        return text

replacements = {
    ' <unk> at <time>':'',
    '<unk>': '',
    '<time>': '',
    '<num>-<num>':'<num>',
    ' <num>.':'',
    'dr.': 'dr',
    '..': '.',
    '. .': '.',
    'a.m.': '',
    'p.m.': '',
    ', , , ,': '',
    
    # Add more replacements as needed
}



## For Testing
a = df['Findings'][13]
print(" \n Before Preprocessing Findings: ", a)

a = df['Impression'][13]
print(" \n Before Preprocessing Impression: ", a)


b = df['File_Path'][0]
print("Before Preprocessing: ", b)

image_paths = df['File_Path']

df['File_Path'] = df['File_Path'].apply(lambda x: x[17:])

b = df['File_Path'][0]
print("After Preprocessing: ", b)

# Remove leading full stops from the 'report' column
df['Findings'] = df['Findings'].apply(lambda x: x.lstrip('.') if isinstance(x, str) and x.startswith('.') else x)
df['Findings'] = df['Findings'].apply(lambda x: x.lstrip('<num>.') if isinstance(x, str) and x.startswith('<num>.') else x)


df['Impression'] = df['Impression'].apply(lambda x: x.lstrip('.') if isinstance(x, str) and x.startswith('.') else x)
df['Impression'] = df['Impression'].apply(lambda x: x.lstrip('<num>.') if isinstance(x, str) and x.startswith('<num>.') else x)

# Perform the series of replacements
for old_value, new_value in replacements.items():
    df['Findings'] = df['Findings'].str.replace(old_value, new_value)
    df['Impression'] = df['Impression'].str.replace(old_value, new_value)
    
df['Findings'] = df['Findings'].apply(remove_sentence)
df['Impression'] = df['Impression'].apply(remove_sentence)

df = df.replace('', pd.NA).dropna()
df = df.reset_index(drop=True)


## For Testing
b = df['Findings'][13]
print(" \n After Preprocessing Findings : ", b)

b = df['Impression'][13]
print(" \n Before Preprocessing Impression: ", b)

print(" \n Length of dataset: ", len(df))

df.to_csv('/home/chayan/MIMIC-Dataset/Data/Final_MIMIC_Dataset_Findings_Impression.csv', index=False)


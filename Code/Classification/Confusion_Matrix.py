import torch
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support


# Load data from CSV file
def load_data_from_csv(csv_file):
    data = pd.read_csv(csv_file)
    actual_labels = data['Binary_Label'].astype(int)    #[0:500]
    predicted_labels = data['Predicted_Label'].astype(int)   #[0:500]
    return actual_labels, predicted_labels
    
    
def calculate_metrics(actual_labels, predicted_labels):
    accuracy = accuracy_score(actual_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='binary')
    
    # Micro-average metrics
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='micro')
    
    # Macro-average metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(actual_labels, predicted_labels, average='macro')
    
    return accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1

# Example usage
if __name__ == "__main__":
    csv_file = "/home/chayan/MIMIC-Dataset/Data/Mimic-test-set-prediction_LRS_linear.csv" # Replace 'labels.csv' with your CSV file containing actual and predicted labels
    actual_labels, predicted_labels = load_data_from_csv(csv_file)
    
    # Calculate confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels)
    
    # Calculate metrics
    accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1 = calculate_metrics(actual_labels, predicted_labels)
    
    # Present results in a tabular form
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Micro Precision', 'Micro Recall', 'Micro F1-Score', 'Macro Precision', 'Macro Recall', 'Macro F1-Score'],
        'Value': [accuracy, precision, recall, f1, micro_precision, micro_recall, micro_f1, macro_precision, macro_recall, macro_f1]
    })
    
    print("Confusion Matrix:")
    print(cm)
    print("\nMetrics:")
    print(results)
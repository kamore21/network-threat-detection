import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model performance using various metrics.
    
    Parameters:
    y_true : list
        Actual labels
    y_pred : list
        Predicted labels
    
    Returns:
    dict
        Dictionary containing accuracy, precision, recall, and F1 score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    return metrics

# Example usage:
# y_true = [0, 1, 0, 1]
# y_pred = [0, 0, 1, 1]
# print(evaluate_model(y_true, y_pred))
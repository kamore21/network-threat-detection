# Data Preprocessing Utilities

## Importing Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

## Data Preprocessing Functions

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean the DataFrame by removing NaN values and duplicates."""
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def encode_labels(df, columns):
    """Encode categorical labels in the specified columns."""
    le = LabelEncoder()
    for column in columns:
        df[column] = le.fit_transform(df[column])
    return df


def scale_features(df):
    """Scale the feature values using StandardScaler."""
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_scaled

# Example usage:
# data = load_data('data.csv')
# cleaned_data = clean_data(data)
# encoded_data = encode_labels(cleaned_data, ['category_column'])
# scaled_data = scale_features(encoded_data)
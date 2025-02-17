import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups

# Fetch the dataset
newsgroups = fetch_20newsgroups(subset='all')

# Data and Labels
X = newsgroups.data
y = newsgroups.target

# Convert the dataset to a DataFrame for easier inspection
df = pd.DataFrame({
    'text': X,
    'label': y
})

# Initialize the information list
info = []

# Basic Dataset Information
info.append("Dataset Information:")
info.append("="*40)

# Number of samples and features
info.append(f"1. Number of Samples (documents): {len(X)}")
info.append(f"2. Number of Features (categories): {len(newsgroups.target_names)}")
info.append(f"3. Categories (Features):")
info.append(f"   - {', '.join(newsgroups.target_names)}")

# First few samples and labels
info.append("\n4. First Few Samples:")
info.append(df.head().to_string(index=False))

# Data Types
info.append("\n5. Data Types of Columns:")
info.append(f"   - Text Column: {df['text'].dtype}")
info.append(f"   - Label Column: {df['label'].dtype}")

# Summary of the text data
info.append("\n6. Text Data Summary:")
info.append(f"   - Number of unique documents: {df['text'].nunique()}")
info.append(f"   - Average document length (in characters): {df['text'].apply(len).mean():.2f}")
info.append(f"   - Maximum document length (in characters): {df['text'].apply(len).max()}")
info.append(f"   - Minimum document length (in characters): {df['text'].apply(len).min()}")

# Basic Statistics on Labels
info.append("\n7. Label Distribution (Class Counts):")
label_counts = df['label'].value_counts().sort_index()
for label, count in label_counts.items():
    info.append(f"   - Category '{newsgroups.target_names[label]}': {count} samples")

# Write the information to a text file
with open("dataset_info.txt", "w") as file:
    for line in info:
        file.write(line + "\n")

print("Dataset information has been saved to 'dataset_info.txt'.")

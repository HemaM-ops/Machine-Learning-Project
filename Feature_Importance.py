from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from Excel
df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")  # Replace "your_dataset.xlsx" with the actual file path



# 'df' is your DataFrame and 'Label' is the name of the target variable column
class_distribution = df["Label"].value_counts()

# Assuming 'target_column' is the name of the column containing the target labels
X = df.drop(columns=['Label'])  # Features (excluding the target column)
y = df['Label']  # Target labels

# 'df' is your DataFrame and 'Label' is the name of the target variable column
class_distribution = df["Label"].value_counts()




# Initialize Random Forest classifier
rf = RandomForestClassifier(random_state=42)

# Fit the model to your data
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_

# Sort feature importances in descending order
sorted_indices = feature_importances.argsort()[::-1]

# Print feature importances
for i, idx in enumerate(sorted_indices):
    print(f"Feature {i+1}: {X.columns[idx]} - Importance: {feature_importances[idx]}")

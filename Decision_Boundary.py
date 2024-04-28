import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from Excel
df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")  # Replace "your_dataset.xlsx" with the actual file path



# 'df' is your DataFrame and 'Label' is the name of the target variable column
class_distribution = df["Label"].value_counts()

# Assuming 'target_column' is the name of the column containing the target labels
X = df.drop(columns=['Label'])  # Features (excluding the target column)
y = df['Label']  # Target labels

# Visualize the dataset
plt.figure(figsize=(10, 8))

# Plot each class separately
for class_label in y.unique():
    X_class = X[y == class_label]
    plt.scatter(X_class.iloc[:, 0], X_class.iloc[:, 1], label=f'Class {class_label}', alpha=0.5)

plt.title('Dataset Visualization (All Features)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()

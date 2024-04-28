import pandas as pd
from sklearn.decomposition import PCA

# Load the training and testing datasets
train_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")
test_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")

# Assuming 'target_column' is the name of the column containing the target labels
X_train = train_df.drop(columns=['Label'])  # Features for training
y_train = train_df['Label']  # Target labels for training

X_test = test_df.drop(columns=['Label'])  # Features for testing
y_test = test_df['Label']  # Target labels for testing

# Initialize PCA models
pca_train = PCA(n_components=95)  # Reduce to 2 dimensions for visualization
pca_test = PCA(n_components=95)

# Fit PCA models to training and testing data separately
X_train_pca = pca_train.fit_transform(X_train)
X_test_pca = pca_test.fit_transform(X_test)

# Convert reduced datasets to DataFrame
train_pca_df = pd.DataFrame(data=X_train_pca, columns=['PC1', 'PC2'])
test_pca_df = pd.DataFrame(data=X_test_pca, columns=['PC1', 'PC2'])

# Add target columns back to the datasets
train_pca_df['target_column'] = y_train
test_pca_df['target_column'] = y_test

# Save reduced datasets to separate output files
train_pca_df.to_excel("train_dataset_pca.xlsx", index=False)
test_pca_df.to_excel("test_dataset_pca.xlsx", index=False)

print("Dimensionality reduction completed and saved to train_dataset_pca.xlsx and test_dataset_pca.xlsx.")

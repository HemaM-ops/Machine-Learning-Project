import pandas as pd
from sklearn.manifold import TSNE

# Load the training and testing datasets
train_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Abstractive_Embeddings_Fasttext_Tamil.xlsx")
test_df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\TestSet_Abstractive_Tamil.xlsx")

# Assuming 'target_column' is the name of the column containing the target labels
X_train = train_df.drop(columns=['Label'])  # Features for training
y_train = train_df['Label']  # Target labels for training

X_test = test_df.drop(columns=['Label'])  # Features for testing
y_test = test_df['Label']  # Target labels for testing

# Initialize t-SNE models
tsne_train = TSNE(n_components=50)  # Reduce to 50 dimensions
tsne_test = TSNE(n_components=50)

# Fit t-SNE models to training and testing data separately
X_train_tsne = tsne_train.fit_transform(X_train)
X_test_tsne = tsne_test.fit_transform(X_test)

# Convert reduced datasets to DataFrame
train_tsne_df = pd.DataFrame(data=X_train_tsne)
test_tsne_df = pd.DataFrame(data=X_test_tsne)

# Add target columns back to the datasets
train_tsne_df['target_column'] = y_train
test_tsne_df['target_column'] = y_test

# Save reduced datasets to separate output files
train_tsne_df.to_excel("train_dataset_tsne.xlsx", index=False)
test_tsne_df.to_excel("test_dataset_tsne.xlsx", index=False)

print("t-SNE dimensionality reduction completed and saved to train_dataset_tsne.xlsx and test_dataset_tsne.xlsx.")

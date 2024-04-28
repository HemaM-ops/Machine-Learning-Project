
#Deleting rows with all zeros
import pandas as pd

# Read the Excel file into a DataFrame
df = pd.read_excel("C:\\Users\\mahad\\Downloads\\ML\\Extractive_Embeddings_Fasttext_Tamil.xlsx")


# Drop rows where all columns have zero values
df = df.loc[(df != 0).any(axis=1)]

# Reset the index if needed
df.reset_index(drop=True, inplace=True)

# Save the modified DataFrame to a new Excel file
output_file = "Missing_Handled_Extractive_Tamil.xlsx"
df.to_excel(output_file, index=False)

print("Modified data saved to", output_file)



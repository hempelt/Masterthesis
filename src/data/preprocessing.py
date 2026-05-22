import pandas as pd
from sklearn.model_selection import train_test_split


#load data from BindingDB_BindingDB_Articles.tsv with unicode encoding
df = pd.read_csv(
    'data/raw/BindingDB_BindingDB_Articles.tsv', 
    sep='\t', 
    encoding='utf-8',
    usecols=['Ligand SMILES', 'BindingDB Target Chain Sequence 1', 'IC50 (nM)'],
    low_memory=False)

#General rule-based preprocessing steps such as removing missing or invalid values and applying the fixed pIC50 transformation 
# were performed before the train-test split, as these steps do not learn information from the data.

# drop all samples with empty values in the columns IC50 (nM), BindingDB Target Chain Sequence 1 and Ligand SMILES
df = df[['Ligand SMILES', 'BindingDB Target Chain Sequence 1', 'IC50 (nM)']].dropna()
#transform column IC50 (nM) to numeric and remove all samples with non-numeric values in the column IC50 (nM)
df['IC50 (nM)'] = pd.to_numeric(df['IC50 (nM)'], errors='coerce')
df = df.dropna(subset=['IC50 (nM)'])
#filter all samples were IC50 is < 0 or > 1e7
df = df[(df['IC50 (nM)'] > 0) & (df['IC50 (nM)'] <= 1e7)]
# Reduce the dataset to 1000 random samples for faster training this has to be removed for the final model training
#df = df.sample(n=1000, random_state=42).reset_index(drop=True)

# For the following steps the data set is split into a training and test set.

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)
# Save train und test data as seperate CSV files
train_df.to_csv('data/processed/train_data.csv', index=False)
test_df.to_csv('data/processed/test_data.csv', index=False)
# Print success message
print(f'✅ Data set was splitted. Train data size: {train_df.shape}, Test data size: {test_df.shape}')

# After splitting, any data-dependent transformations or learned representations should be fitted only on the training set 
# to avoid information leakage from the test set.
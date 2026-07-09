import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import torch


# SMILES-Zeichen-Wörterbuch aus der Arbeit
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}

# Protein-Aminosäuren-Wörterbuch
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}

def label_smiles(smiles, max_len=100):
    encoded = np.zeros(max_len, dtype=np.int64)

    for i, char in enumerate(smiles[:max_len]):
        encoded[i] = CHARISOSMISET.get(char, 0)

    return encoded

def label_protein(sequence, max_len=1200):
    encoded = np.zeros(max_len, dtype=np.int64)

    for i, char in enumerate(sequence[:max_len]):
        encoded[i] = CHARPROTSET.get(char, 0)

    return encoded



def encode_and_save(input_csv, output_path):
    """
    Encodes SMILES and protein sequences for the model
    and saves the resulting tensors as a .pt file.
    """

    # Load data
    df = pd.read_csv(input_csv, sep=',', encoding='utf-8')

    # Label encoding for SMILES strings
    df['smiles_encoded'] = df['Ligand SMILES'].apply(label_smiles)

    # Label encoding for protein sequences
    df['protein_encoded'] = df['BindingDB Target Chain Sequence 1'].apply(label_protein)


    # Prepare arrays
    X_smiles = np.stack(df['smiles_encoded'].values)
    X_protein = np.stack(df['protein_encoded'].values)
    y = df['pIC50'].values.astype(np.float32)

    # Create PyTorch tensors
    X_smiles_tensor = torch.tensor(X_smiles, dtype=torch.long)
    X_protein_tensor = torch.tensor(X_protein, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)


    # Save tensors
    torch.save(
        {
            "X_smiles": X_smiles_tensor,
            "X_protein": X_protein_tensor,
            "y": y_tensor
        },
        output_path
    )

    print(f"Encoded data saved to: {output_path}")
    print("X_smiles:", X_smiles_tensor.shape)
    print("X_protein:", X_protein_tensor.shape)
    print("y:", y_tensor.shape)


encode_and_save(
    "data/processed/train_data.csv",
    "data/encoded/encoded_train.pt"
)

encode_and_save(
    "data/processed/test_data.csv",
    "data/encoded/encoded_test.pt"
)
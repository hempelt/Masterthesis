import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path


# Protein-Aminosäuren-Wörterbuch wie bei AttentionDTA
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}


# ------------------------------------------------------------
# Protein-Encoding bleibt unverändert
# ------------------------------------------------------------

def label_protein(sequence, max_len=1200):
    """
    Encodiert eine Proteinsequenz zeichenweise mit dem gleichen
    CHARPROTSET-Wörterbuch wie im AttentionDTA-Baseline-Modell.
    """
    encoded = np.zeros(max_len, dtype=np.int64)

    for i, char in enumerate(sequence[:max_len]):
        encoded[i] = CHARPROTSET.get(char, 0)

    return encoded




# ------------------------------------------------------------
# SMILES-8-mer-Encoding nach WideDTA-Idee
# ------------------------------------------------------------

def SMILES_to_kmers(sequence, k=8):
    """
    Zerlegt eine SMILES-Sequenz in überlappende 8-mer-Wörter.

    Beispiel:
    CCOCOOOCCC -> CCOCOOOC, COCOOOCC, OCOOOCCC

    """
    sequence = str(sequence)

    if len(sequence) < k:
        return []

    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def build_SMILES_kmer_vocab(smiles_sequences, k=8):
    """
    Erstellt ein Vocabulary für SMILES-k-mers aus den Trainingssequenzen.

    0 = Padding
    1 = Unknown Token für k-mers, die im Testset vorkommen,
        aber im Trainingsvokabular nicht enthalten sind.
    """
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for sequence in smiles_sequences:
        kmers = SMILES_to_kmers(sequence, k=k)

        for kmer in kmers:
            if kmer not in vocab:
                vocab[kmer] = len(vocab)

    return vocab


def label_SMILES_kmers(sequence, vocab, max_len=100, k=8):
    """
    Encodiert eine SMILES-Sequenz als Sequenz von k-mer-IDs.

    Die Ausgabe hat eine feste Länge von max_len:
    - längere Sequenzen werden abgeschnitten
    - kürzere Sequenzen werden mit 0 gepaddet
    """
    encoded = np.zeros(max_len, dtype=np.int64)

    kmers = SMILES_to_kmers(sequence, k=k)

    for i, kmer in enumerate(kmers[:max_len]):
        encoded[i] = vocab.get(kmer, vocab["<UNK>"])

    return encoded


# ------------------------------------------------------------
# Encoding-Funktion für Train und Test
# ------------------------------------------------------------

def encode_and_save(input_csv, output_path, vocab, k=8):
    """
    Encodiert SMILES als überlappende 8-mer-Wörter und Proteinsequenzen
    zeichenweise wie in der AttentionDTA-Baseline.
    Anschließend werden die Tensoren als .pt-Datei gespeichert.
    """

    # Daten laden
    df = pd.read_csv(input_csv, sep=",", encoding="utf-8")

    # Protein-Encoding bleibt identisch zur AttentionDTA-Baseline
    df["protein_encoded"] = df['BindingDB Target Chain Sequence 1'].apply(label_protein)

    # SMILES-Encoding als 8-mer-Wörter
    df["smiles_encoded"] = df["Ligand SMILES"].apply(
        lambda seq: label_SMILES_kmers(
            sequence=seq,
            vocab=vocab,
            max_len=100,
            k=k
        )
    )

    # IC50 in pIC50 transformieren
    df["pIC50"] = -np.log10(df["IC50 (nM)"].astype(float) * 1e-9)

    # Arrays vorbereiten
    X_smiles = np.stack(df["smiles_encoded"].values)
    X_protein = np.stack(df["protein_encoded"].values)
    y = df["pIC50"].values.astype(np.float32)

    # PyTorch-Tensoren erstellen
    X_smiles_tensor = torch.tensor(X_smiles, dtype=torch.long)
    X_protein_tensor = torch.tensor(X_protein, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    # Tensoren speichern
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


# ------------------------------------------------------------
# Pfade
# ------------------------------------------------------------

train_csv = "data/processed/train_data.csv"
test_csv = "data/processed/test_data.csv"

encoded_train_path = "data/encoded/encoded_train_smiles_8mer.pt"
encoded_test_path = "data/encoded/encoded_test_smiles_8mer.pt"

vocab_path = "data/encoded/smiles_8mer_vocab.json"

Path("data/encoded").mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Vocabulary nur auf Trainingsdaten erstellen
# ------------------------------------------------------------

train_df = pd.read_csv(train_csv, sep=",", encoding="utf-8")

smiles_vocab = build_SMILES_kmer_vocab(
    smiles_sequences=train_df["Ligand SMILES"],
    k=8
)

print("SMILES 8-mer vocabulary size:", len(smiles_vocab))

# Vocabulary speichern, damit später exakt dasselbe Mapping nachvollziehbar ist
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(smiles_vocab, f, indent=2)

print(f"SMILES vocabulary saved to: {vocab_path}")


# ------------------------------------------------------------
# Train- und Testdaten encodieren
# ------------------------------------------------------------

encode_and_save(
    input_csv=train_csv,
    output_path=encoded_train_path,
    vocab=smiles_vocab,
    k=8
)

encode_and_save(
    input_csv=test_csv,
    output_path=encoded_test_path,
    vocab=smiles_vocab,
    k=8
)
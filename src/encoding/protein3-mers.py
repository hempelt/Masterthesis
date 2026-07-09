import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path


# ------------------------------------------------------------
# SMILES-Zeichen-Wörterbuch wie bei AttentionDTA
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# SMILES-Encoding bleibt unverändert
# ------------------------------------------------------------

def label_smiles(smiles, max_len=100):
    """
    Encodiert einen SMILES-String zeichenweise mit dem gleichen
    CHARISOSMISET-Wörterbuch wie im AttentionDTA-Baseline-Modell.
    """
    encoded = np.zeros(max_len, dtype=np.int64)

    for i, char in enumerate(str(smiles)[:max_len]):
        encoded[i] = CHARISOSMISET.get(char, 0)

    return encoded


# ------------------------------------------------------------
# Protein-3-mer-Encoding nach WideDTA-Idee
# ------------------------------------------------------------

def protein_to_kmers(sequence, k=3):
    """
    Zerlegt eine Proteinsequenz in überlappende k-mer-Wörter.

    Beispiel:
    MTVKTE -> MTV, TVK, VKT, KTE
    """
    sequence = str(sequence)

    if len(sequence) < k:
        return []

    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]


def build_protein_kmer_vocab(protein_sequences, k=3):
    """
    Erstellt ein Vocabulary für Protein-k-mers aus den Trainingssequenzen.

    0 = Padding
    1 = Unknown Token für k-mers, die im Testset vorkommen,
        aber im Trainingsvokabular nicht enthalten sind.
    """
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for sequence in protein_sequences:
        kmers = protein_to_kmers(sequence, k=k)

        for kmer in kmers:
            if kmer not in vocab:
                vocab[kmer] = len(vocab)

    return vocab


def label_protein_kmers(sequence, vocab, max_len=1200, k=3):
    """
    Encodiert eine Proteinsequenz als Sequenz von k-mer-IDs.

    Die Ausgabe hat eine feste Länge von max_len:
    - längere Sequenzen werden abgeschnitten
    - kürzere Sequenzen werden mit 0 gepaddet
    """
    encoded = np.zeros(max_len, dtype=np.int64)

    kmers = protein_to_kmers(sequence, k=k)

    for i, kmer in enumerate(kmers[:max_len]):
        encoded[i] = vocab.get(kmer, vocab["<UNK>"])

    return encoded


# ------------------------------------------------------------
# Encoding-Funktion für Train und Test
# ------------------------------------------------------------

def encode_and_save(input_csv, output_path, protein_vocab, k=3):
    """
    Encodiert SMILES wie bisher zeichenweise und Proteinsequenzen
    als WideDTA-artige 3-mer-Wörter. Anschließend werden die Tensoren
    als .pt-Datei gespeichert.
    """

    # Daten laden
    df = pd.read_csv(input_csv, sep=",", encoding="utf-8")

    # SMILES-Encoding bleibt identisch zur AttentionDTA-Baseline
    df["smiles_encoded"] = df["ligand_smiles"].apply(label_smiles)

    # Protein-Encoding als 3-mer-Wörter
    df["protein_encoded"] = df["protein_sequence"].apply(
        lambda seq: label_protein_kmers(
            sequence=seq,
            vocab=protein_vocab,
            max_len=1200,
            k=k
        )
    )


    # Arrays vorbereiten
    X_smiles = np.stack(df["smiles_encoded"].values)
    X_protein = np.stack(df["protein_encoded"].values)
    y = df["pic50"].values.astype(np.float32)

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

encoded_train_path = "data/encoded/encoded_train_protein_3mer.pt"
encoded_test_path = "data/encoded/encoded_test_protein_3mer.pt"

vocab_path = "data/encoded/protein_3mer_vocab.json"

Path("data/encoded").mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Vocabulary nur auf Trainingsdaten erstellen
# ------------------------------------------------------------

train_df = pd.read_csv(train_csv, sep=",", encoding="utf-8")

protein_vocab = build_protein_kmer_vocab(
    protein_sequences=train_df["protein_sequence"],
    k=3
)

print("Protein 3-mer vocabulary size:", len(protein_vocab))

# Vocabulary speichern, damit später exakt dasselbe Mapping nachvollziehbar ist
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(protein_vocab, f, indent=2)

print(f"Protein vocabulary saved to: {vocab_path}")


# ------------------------------------------------------------
# Train- und Testdaten encodieren
# ------------------------------------------------------------

encode_and_save(
    input_csv=train_csv,
    output_path=encoded_train_path,
    protein_vocab=protein_vocab,
    k=3
)

encode_and_save(
    input_csv=test_csv,
    output_path=encoded_test_path,
    protein_vocab=protein_vocab,
    k=3
)
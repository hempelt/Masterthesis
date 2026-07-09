import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Datensatz aus BindingDB_BindingDB_Articles.tsv mit UTF-8-Encoding laden
df = pd.read_csv('data/raw/BindingDB_BindingDB_Articles.tsv', sep='\t', encoding='utf-8', usecols=['Ligand SMILES', 'BindingDB Target Chain Sequence 1', 'IC50 (nM)'])

# Spaltennamen bereinigen: Leerzeichen entfernen, in Kleinbuchstaben umwandeln und Leerzeichen durch Unterstriche ersetzen
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
# Entferne Sonderzeichen aus den Spaltennamen
df.columns = df.columns.str.replace(r"[^a-z0-9_]", "", regex=True)
# Spalte bindingdb_target_chain_sequence_1 in protein_sequence umbenennen
df.rename(columns={'bindingdb_target_chain_sequence_1': 'protein_sequence'}, inplace=True)

# Alle Datensätze mit fehlenden Werten in den Spalten ic50_nm, protein_sequence und ligand_smiles entfernen
df = df[['ligand_smiles', 'protein_sequence', 'ic50_nm']].dropna()

# Spalte ic50_nm in numerische Werte umwandeln. # Nicht-numerische Werte werden zu NaN und anschließend entfernt
df['ic50_nm'] = pd.to_numeric(df['ic50_nm'], errors='coerce')
df = df.dropna(subset=['ic50_nm'])

# Duplikate basierend auf den Spalten ligand_smiles, protein_sequence und ic50_nm entfernen
df = df.drop_duplicates(subset=["ligand_smiles", "protein_sequence", "ic50_nm"],keep="first")

# IC50 muss positiv sein, da log10 nur für positive Werte definiert ist
df = df[df["ic50_nm"] > 0]

# IC50-Werte von nM in Molar (M) umrechnen
IC50_M = df["ic50_nm"] * 1e-9

# IC50-Werte durch negative dekadische Logarithmierung in pIC50 transformieren
df["pic50"] = -np.log10(IC50_M)

# die Spalte ic50nm entfernen, da wir nur noch die Spalte pic50 benötigen
df = df.drop(columns=["ic50_nm"])

# Datensatz in Trainings- und Testdaten aufteilen
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Trainings- und Testdaten als separate CSV-Dateien speichern
train_df.to_csv('data/processed/train_data.csv', index=False)
test_df.to_csv('data/processed/test_data.csv', index=False)

# Erfolgsmeldung mit Größe der Trainings- und Testdaten ausgeben
print(f'✅ Data set was split. Train data size: {train_df.shape}, Test data size: {test_df.shape}')

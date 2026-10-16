# Masterthesis – Vergleichende Analyse sequenzbasierter Eingaberepräsentationen und Deep-Learning Architekturen zur Vorhersage von Ligand-Protein Bindungsaffinitäten

## Projektbeschreibung

Dieses Repository enthält den im Rahmen der Masterarbeit entwickelten Python-Code zur Untersuchung sequenzbasierter Eingaberepräsentationen und Deep-Learning-Modellvarianten für die Vorhersage von Ligand-Protein-Bindungsaffinitäten.

Ziel der Arbeit ist es, den Einfluss unterschiedlicher Eingaberepräsentationen von Proteinen und Liganden sowie verschiedener Deep-Learning-Architekturen auf die Vorhersageleistung eines Drug-Target-Affinity-Modells zu untersuchen.

Als Ausgangsmodell dient eine AttentionDTA-basierte Architektur nach Zhao et al. (2023). Das Modell wird auf experimentell bestimmten Ligand-Protein-Affinitätsdaten aus der BindingDB-Datenbank trainiert und anschließend durch verschiedene Encoding-Verfahren und Architekturkomponenten erweitert.

---

## Datengrundlage

Als Datengrundlage werden experimentell bestimmte Ligand-Protein-Affinitätsdaten aus BindingDB verwendet.

Die BindingDB Rohdaten befinden sich im Verzeichnis `data/raw/BindingDB_BindingDB_Articles.tsv`
Eine Beschreibung des Datensatzes befindet sich ebenfalls im Verzeichnis: `data\raw\Description_BindingDB_TSV_File.pdf`

Für die Modellierung werden folgende Informationen genutzt:

- Ligand als SMILES-Sequenz
- Protein als Aminosäuresequenz
- IC50-Wert als experimentelle Zielgröße

---

## EDA Explorative Datenanalyse




## Datenvorverarbeitung

Die Datenvorverarbeitung umfasst unter anderem:

- Auswahl der relevanten Spalten
- Entfernung fehlender Werte
- Konvertierung der IC50-Werte in numerische Werte
- Filterung ungültiger bzw. sehr hoher IC50-Werte
- Entfernung exakter Duplikate
- Transformation von IC50 zu pIC50
- Aufteilung in Trainings- und Testdaten

Alle regelbasierten Transformationen wurden vor dem Train-Test-Split durchgeführt. Datenabhängige Transformationen, beispielsweise die Erstellung von k-mer-Vokabularen, wurden ausschließlich auf Basis der Trainingsdaten durchgeführt, um Data Leakage zu vermeiden.

Der Quellcode für die Datenvorverarbeitung befindet sich im Skript `src/data/preprocessing.py`.
Die verarbeiteten Trainings- und Testdatensätze werden in das Verzeichnis `data\processed` gespeichert

---

## Sequenzbasierte Eingaberepräsentationen

Der Quellcode für die untersuchten sequenzbasierten Eingaberepräsentationen befindet sich im Verzeichnis `src/encoding/`. Für jedes Encoding-Verfahren wurde eine eigene Python-Datei erstellt.

### Baseline-Encoding
Datei: `src/encoding/AttentionDTA_Baseline.py`

Zeichenbasiertes Encoding von:
- SMILES-Sequenzen
- Proteinsequenzen

### Protein-3-mer-Encoding
Datei: `src/encoding/protein3-mers.py`

Die Proteinsequenz wird in überlappende Teilsequenzen aus jeweils drei Aminosäuren zerlegt und anschließend numerisch codiert.

### SMILES-8-mer-Encoding
Datei: `src/encoding/SMILES8-mers.py`

Die SMILES-Sequenz wird in überlappende Teilsequenzen aus jeweils acht Zeichen zerlegt und anschließend numerisch codiert.

### Kombiniertes Encoding
Datei: `src/encoding/protein3-SMILE8-mers.py`

Kombination aus:
- Protein-3-mer-Encoding
- SMILES-8-mer-Encoding

Für das Training der Modelle wurden die erzeugten die erzeugten .pt-Dateien mit den codierten Eingabedaten im Verzeichnis `data\encoded` gespeichert und anschließend in Google Drive hochgeladen und danach in Google Colab für das Training und die Evaluation der Modelle geladen.
---

## Modellvarianten

Der Quellcode für die untersuchten Deep-Learning-Modellvarianten befindet sich im Verzeichnis `src/models/`. Für jede Modellvariante wurde eine eigene Python-Datei erstellt.

### AttentionDTA
Datei: `src/models/attention_dta.py`

Implementierung des AttentionDTA-basierten Ausgangsmodells.

### AttentionDTA-BiLSTM
Datei: `src/models/attention_dta_bilstm.py`

Erweiterung des Ausgangsmodells um eine bidirektionale LSTM-Schicht zur zusätzlichen Verarbeitung der Proteinsequenz.

### AttentionDTA-Inception
Datei: `src/models/attention_dta_inception.py`

Erweiterung des Ausgangsmodells um Inception-Blöcke mit parallelen 1D-Convolutional-Schichten unterschiedlicher Kernelgrößen.

### AttentionDTA-Inception-BiLSTM
Datei: `src/models/attention_dta_inception_bilstm.py`

Kombination der BiLSTM- und Inception-Erweiterung innerhalb der AttentionDTA-basierten Modellarchitektur. Für das Training der Modelle in Google Colab wurden die Python Dateien in Google Drive hochgeladen.

---

## Training

Das Training der Deep-Learning-Modelle wurde aufgrund des erforderlichen GPU-Rechenaufwands in Google Colab durchgeführt.

Das für die Experimente verwendete Notebook befindet sich im Repository unter:

`notebooks\training_evaluation_colab.ipynb`

Das Notebook enthält teilweise Google-Colab-spezifische Komponenten, insbesondere:

- Einbindung von Google Drive
- absolute Google-Drive-Pfade
- GPU-spezifische Trainingskonfigurationen

Diese Bestandteile beziehen sich ausschließlich auf die verwendete Rechenumgebung und sind nicht Bestandteil der eigentlichen Modelllogik.

Die Modellarchitekturen, Trainingsschritte und Evaluationsverfahren sind im Notebook vollständig dokumentiert.

---

## Trainingsparameter

Für die untersuchten Modellvarianten wurden weitgehend einheitliche Trainingsparameter verwendet:

- Loss Function: Mean Squared Error
- Optimizer: AdamW
- Learning Rate: 5 × 10^-5
- Weight Decay: 1 × 10^-4
- Batch Size: 128
- Maximale Epochenzahl: 500
- Early Stopping: Patience 50
- Random Seed: 4321

Zur Steuerung der Lernrate wurde zusätzlich ein Cyclic Learning Rate Scheduler verwendet.

---

## Evaluation

Die Modellleistung wird anhand folgender Regressionsmetriken bewertet:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Bestimmtheitsmaß R²

Die Metriken werden direkt auf der pIC50-Skala berechnet.

Der während des Trainings verwendete Validierungsdatensatz dient ausschließlich der Modellauswahl und dem Early Stopping.

Die finale Bewertung der Modellleistung erfolgt anhand eines zuvor zurückgehaltenen Testdatensatzes.

Die Ergebnisse der Evaluation befinden sich ebenfalls in der Datei `notebooks\training_evaluation_colab.ipynb`

---

## Repository-Struktur

```text
Masterthesis/
│
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── eda/
│   ├── preprocessing/
│   ├── encoding/
│   └── training/
│       └── training_evaluation.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
└── models/
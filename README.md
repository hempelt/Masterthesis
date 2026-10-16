# Masterthesis – Vergleichende Analyse sequenzbasierter Eingaberepräsentationen und Deep-Learning-Architekturen zur Vorhersage von Ligand-Protein-Bindungsaffinitäten

## Projektbeschreibung

Dieses Repository enthält den im Rahmen der Masterarbeit entwickelten Python-Code zur Untersuchung sequenzbasierter Eingaberepräsentationen und Deep-Learning-Modellvarianten für die Vorhersage von Ligand-Protein-Bindungsaffinitäten.

Ziel der Arbeit ist es, den Einfluss unterschiedlicher Eingaberepräsentationen von Proteinen und Liganden sowie verschiedener Deep-Learning-Architekturen auf die Vorhersageleistung eines Drug-Target-Affinity-Modells zu untersuchen.

Als Ausgangsmodell dient eine AttentionDTA-basierte Architektur nach Zhao et al. (2023). Das Modell wird auf experimentell bestimmten Ligand-Protein-Daten aus der BindingDB-Datenbank trainiert und anschließend durch verschiedene Encoding-Verfahren und Architekturkomponenten erweitert.

---

## Datengrundlage

Als Datengrundlage werden experimentell bestimmte Ligand-Protein-Daten aus BindingDB verwendet. Für die Modellierung werden IC50-Werte als quantitative Zielgröße genutzt.

Da die Rohdaten aus BindingDB einen hohen Speicherbedarf aufweisen, wurden diese nicht in das GitHub-Repository aufgenommen.

Um den Code auszuführen, muss zunächst die BindingDB-Datei unter **„TSV file containing data curated by BindingDB from scientific articles“** über die [UC San Diego Library](https://library.ucsd.edu/dc/object/bb54426945) heruntergeladen 

Die heruntergeladene Datei muss anschließend unter folgendem Pfad gespeichert werden:

`data/raw/BindingDB_BindingDB_Articles.tsv`

Eine Beschreibung des verwendeten Datensatzes befindet sich als PDF-Datei unter:

`data/raw/Description_BindingDB_TSV_File.pdf`

Für die Modellierung werden folgende Informationen verwendet:

- Liganden in Form von SMILES-Sequenzen
- Proteine in Form von Aminosäuresequenzen
- IC50-Werte als experimentell bestimmte Zielgröße

---

## Explorative Datenanalyse (EDA)

Die explorative Datenanalyse wurde im Jupyter-Notebook `notebooks/EDA.ipynb` durchgeführt. Ziel der Analyse ist es, die Struktur und Verteilung der verwendeten BindingDB-Daten zu untersuchen und relevante Eigenschaften des Datensatzes vor der Modellierung zu identifizieren.

Im Rahmen der EDA werden unter anderem folgende Aspekte untersucht:

- Verfügbarkeit und Vollständigkeit der Affinitätsmessungen
- Verteilung der IC50-Werte
- Transformation der IC50-Werte in die logarithmische pIC50-Skala
- Verteilung der pIC50-Werte
- Länge der Proteinsequenzen
- Länge der SMILES-Sequenzen
- Anzahl unterschiedlicher Liganden und Proteine
- Häufigkeit von Ligand-Protein-Interaktionen
- Auftreten von Duplikaten und mehrfach vorhandenen Messungen

Die Ergebnisse der explorativen Datenanalyse bilden die Grundlage für die anschließende Datenvorverarbeitung und die Festlegung relevanter Parameter, beispielsweise der maximalen Sequenzlängen und des verwendeten Wertebereichs der Zielvariable.

Das Notebook befindet sich unter:

`notebooks/EDA.ipynb`

Die im Rahmen der explorativen Datenanalyse erzeugten und in der Masterarbeit verwendeten Abbildungen befinden sich im Verzeichnis `reports/`.

---

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

Der Quellcode für die Datenvorverarbeitung befindet sich im Skript:

`src/data/preprocessing.py`

Die verarbeiteten Trainings- und Testdatensätze werden im Verzeichnis `data/processed/` gespeichert. Aufgrund ihres Speicherbedarfs sind diese Dateien nicht Bestandteil des GitHub-Repositories.

---

## Sequenzbasierte Eingaberepräsentationen

Der Quellcode für die untersuchten sequenzbasierten Eingaberepräsentationen befindet sich im Verzeichnis `src/encoding/`. Für jedes Encoding-Verfahren wurde eine eigene Python-Datei erstellt.

### Baseline-Encoding

Datei:

`src/encoding/AttentionDTA_Baseline.py`

Zeichenbasiertes Encoding von:

- SMILES-Sequenzen
- Proteinsequenzen

### Protein-3-mer-Encoding

Datei:

`src/encoding/protein3-mers.py`

Die Proteinsequenz wird in überlappende Teilsequenzen aus jeweils drei Aminosäuren zerlegt und anschließend numerisch codiert.

### SMILES-8-mer-Encoding

Datei:

`src/encoding/SMILES8-mers.py`

Die SMILES-Sequenz wird in überlappende Teilsequenzen aus jeweils acht Zeichen zerlegt und anschließend numerisch codiert.

### Kombiniertes Encoding

Datei:

`src/encoding/protein3-SMILE8-mers.py`

Kombination aus:

- Protein-3-mer-Encoding
- SMILES-8-mer-Encoding

Für das Training der Modelle wurden die erzeugten `.pt`-Dateien mit den codierten Eingabedaten im Verzeichnis `data/encoded/` gespeichert. Danach wurden die Dateien in Google Drive hochgeladen und anschließend in Google Colab für das Training und die Evaluation der Modelle geladen. Aufgrund ihrer Dateigröße wurden diese Dateien nicht in das GitHub-Repository aufgenommen.

Die für die k-mer-basierten Encodings erstellten Vokabulare befinden jedoch als json-Dateien im Verzeichnis `data/encoded/`.

---

## Modellvarianten

Der Quellcode für die untersuchten Deep-Learning-Modellvarianten befindet sich im Verzeichnis `src/models/`. Für jede Modellvariante wurde eine eigene Python-Datei erstellt.

### AttentionDTA

Datei:

`src/models/attention_dta.py`

Implementierung des AttentionDTA-basierten Ausgangsmodells.

### AttentionDTA-BiLSTM

Datei:

`src/models/attention_dta_bilstm.py`

Erweiterung des Ausgangsmodells um eine bidirektionale LSTM-Schicht zur zusätzlichen Verarbeitung der Proteinsequenz.

### AttentionDTA-Inception

Datei:

`src/models/attention_dta_inception.py`

Erweiterung des Ausgangsmodells um Inception-Blöcke mit parallelen eindimensionalen Convolutional-Schichten unterschiedlicher Kernelgrößen.

### AttentionDTA-Inception-BiLSTM

Datei:

`src/models/attention_dta_inception_bilstm.py`

Kombination der BiLSTM- und Inception-Erweiterung innerhalb der AttentionDTA-basierten Modellarchitektur.

Für das Training in Google Colab wurden die Python-Dateien der jeweiligen Modellvarianten in Google Drive abgelegt und von dort in die Colab-Laufzeitumgebung eingebunden.

---

## Training

Das Training der Deep-Learning-Modelle wurde aufgrund des erforderlichen GPU-Rechenaufwands in Google Colab durchgeführt.

Das für die Experimente verwendete Notebook befindet sich im Repository unter:

`notebooks/training_evaluation_colab.ipynb`

Das Notebook enthält teilweise Google-Colab-spezifische Komponenten, insbesondere:

- Einbindung von Google Drive
- absolute Google-Drive-Pfade
- GPU-spezifische Trainingskonfigurationen

Diese Bestandteile beziehen sich ausschließlich auf die verwendete Rechenumgebung und sind nicht Bestandteil der eigentlichen Modelllogik.

Die Modellarchitekturen, Trainingsschritte und Evaluationsverfahren sind im Notebook dokumentiert.

Für die Modellberechnungen wurde in Google Colab eine A100-GPU als Hardwarebeschleuniger verwendet. Zusätzlich wurde die Option für erweiterten Arbeitsspeicher aktiviert.

Die Datenvorverarbeitung und das Encoding der Eingabesequenzen wurden lokal mit Python durchgeführt.

---

## Evaluation

Die Modellleistung wird anhand folgender Regressionsmetriken bewertet:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Bestimmtheitsmaß R²

Die Metriken werden direkt auf der logarithmisch transformierten pIC50-Skala berechnet.

Der während des Trainings verwendete Validierungsdatensatz dient ausschließlich der Modellüberwachung und dem Early Stopping.

Die finale Bewertung der Modellleistung erfolgt anhand eines zuvor zurückgehaltenen Testdatensatzes.

Alle Modelle wurden mit einem festen Random Seed trainiert, um die Vergleichbarkeit der Ergebnisse zu erhöhen. Aufgrund stochastischer Trainingsprozesse und nicht vollständig deterministischer GPU-Operationen können dennoch geringe Abweichungen zwischen wiederholten Trainingsläufen auftreten.

Die Ergebnisse der Evaluation befinden sich ebenfalls im Notebook:

`notebooks/training_evaluation_colab.ipynb`


---

## Installation und Umgebung

Die für das Projekt verwendete Conda-Umgebung ist in der Datei `environment.yml` dokumentiert.

Die Umgebung kann mit folgendem Befehl erstellt werden:

```bash
conda env create -f environment.yml
```

Anschließend kann die Umgebung aktiviert werden:

```bash
conda activate masterthesis
```

Alternativ sind die verwendeten Python-Abhängigkeiten in der Datei `requirements.txt` aufgeführt.

Die Installation der dort aufgeführten Pakete kann mit folgendem Befehl erfolgen:

```bash
pip install -r requirements.txt
```

---

## Repository-Struktur

```text
Masterthesis/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
│
├── data/
│   ├── raw/
│   │   └── BindingDB-Rohdaten und Datensatzbeschreibung
│   ├── processed/
│   │   └── Vorverarbeitete Trainings- und Testdaten
│   └── encoded/
│       └── Codierte Eingabedaten und Vokabulare
│
├── src/
│   ├── data/
│   │   └── preprocessing.py
│   │
│   ├── encoding/
│   │   ├── AttentionDTA_Baseline.py
│   │   ├── protein3-mers.py
│   │   ├── SMILES8-mers.py
│   │   └── protein3-SMILE8-mers.py
│   │
│   └── models/
│       ├── attention_dta.py
│       ├── attention_dta_bilstm.py
│       ├── attention_dta_inception.py
│       └── attention_dta_inception_bilstm.py
│
├── notebooks/
│   ├── EDA.ipynb
│   └── training_evaluation_colab.ipynb
│
└── reports/
    └── Abbildungen und Auswertungen aus EDA und Modellevaluation
```

---

## Hinweise zur Reproduzierbarkeit

Die großen BindingDB-Rohdaten, vorverarbeiteten Datensätze und erzeugten `.pt`-Dateien sind aufgrund ihres Speicherbedarfs nicht Bestandteil des GitHub-Repositories.

Zur vollständigen Reproduktion müssen daher zunächst die BindingDB-Rohdaten heruntergeladen und im vorgesehenen Verzeichnis abgelegt werden. Anschließend können die Datenvorverarbeitung und die verschiedenen Encoding-Verfahren mithilfe der im Repository enthaltenen Python-Skripte ausgeführt werden.

Das eigentliche Modelltraining wurde in Google Colab durchgeführt. Das zugehörige Notebook enthält daher teilweise Google-Colab- und Google-Drive-spezifische Pfade, die bei einer Ausführung in einer anderen Umgebung entsprechend angepasst werden müssen.

---

## Referenz des Ausgangsmodells

Als Ausgangspunkt für die Modellimplementierung wurde die AttentionDTA-Architektur nach Zhao et al. (2023) verwendet.

Die vollständige bibliografische Angabe befindet sich im Literaturverzeichnis der zugehörigen Masterarbeit.
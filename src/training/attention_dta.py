import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import mlflow
import mlflow.pytorch

from src.models.AttentionDTA import AttentionDTA

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from pathlib import Path
import copy
import random
import numpy as np


# ------------------------------------------------------------
# Hyperparameter und Reproduzierbarkeit
# ------------------------------------------------------------

# Lernrate entsprechend dem ursprünglichen AttentionDTA-Training
learning_rate = 5e-5

# Anzahl der Samples pro Batch
batch_size = 128

# Maximale Anzahl an Trainingsepochen
# Durch Early Stopping kann das Training früher abbrechen
epochs = 500

# Weight Decay zur Regularisierung der Modellgewichte
weight_decay = 1e-4

# Fester Seed für reproduzierbare Ergebnisse
seed = 4321

# Seeds für verschiedene Zufallsquellen setzen
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Encodierte Trainings- und Testdaten laden
# ------------------------------------------------------------

# Die Daten wurden zuvor bereits encodiert und als PyTorch-Dateien gespeichert
train_data = torch.load("data/encoded/encoded_train.pt")
test_data = torch.load("data/encoded/encoded_test.pt")

# TensorDataset fasst SMILES-Tensoren, Protein-Tensoren und Zielwerte zusammen
train_dataset = TensorDataset(
    train_data["X_smiles"],
    train_data["X_protein"],
    train_data["y"]
)

test_dataset = TensorDataset(
    test_data["X_smiles"],
    test_data["X_protein"],
    test_data["y"]
)


# ------------------------------------------------------------
# Trainingsdaten in Trainings- und Validierungsset aufteilen
# ------------------------------------------------------------

# 20 % der Trainingsdaten werden als Validierungsset verwendet
# Das unabhängige Testset bleibt unverändert und wird erst am Ende genutzt
valid_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - valid_size

# Reproduzierbarer Train/Validation-Split
train_subset, valid_subset = random_split(
    train_dataset,
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(seed)
)


# ------------------------------------------------------------
# DataLoader erstellen
# ------------------------------------------------------------

# DataLoader erzeugen Mini-Batches für Training, Validierung und Test
train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(seed)
)

valid_loader = DataLoader(
    valid_subset,
    batch_size=batch_size,
    shuffle=False,
    generator=torch.Generator().manual_seed(seed)
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    generator=torch.Generator().manual_seed(seed)
)


# ------------------------------------------------------------
# Gerät auswählen: GPU falls verfügbar, sonst CPU
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Modell initialisieren
# ------------------------------------------------------------

# AttentionDTA-Modell erstellen und auf GPU/CPU verschieben
model = AttentionDTA().to(device)

# Xavier-Initialisierung für alle Gewichtsmatrizen
# Bias-Parameter werden dadurch nicht verändert
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)


# ------------------------------------------------------------
# Loss-Funktion definieren
# ------------------------------------------------------------

# Mean Squared Error wird für die Regression der Bindungsaffinität verwendet
loss_fn = nn.MSELoss()


# ------------------------------------------------------------
# Optimizer vorbereiten: AdamW mit Weight Decay
# ------------------------------------------------------------

# Im Originaltraining wird Weight Decay nur auf Gewichtsmatrizen angewendet,
# nicht auf Bias-Parameter. Daher werden die Parameter hier getrennt.
weight_p = []
bias_p = []

for name, parameter in model.named_parameters():
    if "bias" in name:
        bias_p.append(parameter)
    else:
        weight_p.append(parameter)

# AdamW-Optimizer mit Weight Decay für Gewichte
# Bias-Parameter erhalten keinen Weight Decay
optimizer = torch.optim.AdamW(
    [
        {"params": weight_p, "weight_decay": weight_decay},
        {"params": bias_p, "weight_decay": 0}
    ],
    lr=learning_rate
)


# ------------------------------------------------------------
# Learning Rate Scheduler: CyclicLR
# ------------------------------------------------------------

# Schrittweite für den zyklischen Lernratenverlauf
# Entspricht ungefähr einer halben Zykluslänge bezogen auf die Trainingsbatches
step_size_up = max(1, train_size // batch_size)

# CyclicLR variiert die Lernrate zwischen base_lr und max_lr
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=learning_rate,
    max_lr=learning_rate * 10,
    cycle_momentum=False,
    step_size_up=step_size_up
)


# ------------------------------------------------------------
# MLflow konfigurieren
# ------------------------------------------------------------

# Speicherort für lokale MLflow-Runs
mlruns_path = Path("mlruns").resolve()

# Name des MLflow-Runs
run_name = "AttentionDTA_Baseline"

# MLflow Tracking URI setzen
mlflow.set_tracking_uri(mlruns_path.as_uri())

# Experimentname in MLflow
mlflow.set_experiment("dta_model_comparison")


# ------------------------------------------------------------
# Training mit MLflow-Logging starten
# ------------------------------------------------------------

with mlflow.start_run(run_name=run_name):

    # --------------------------------------------------------
    # Hyperparameter und Modellkonfiguration in MLflow loggen
    # --------------------------------------------------------

    mlflow.log_param("model", "AttentionDTA")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("weight_decay", weight_decay)

    mlflow.log_param("scheduler", "CyclicLR")
    mlflow.log_param("base_lr", learning_rate)
    mlflow.log_param("max_lr", learning_rate * 10)
    mlflow.log_param("step_size_up", step_size_up)

    mlflow.log_param("weight_decay_bias", 0)

    mlflow.log_param("smiles_max_len", 100)
    mlflow.log_param("protein_max_len", 1200)
    mlflow.log_param("embedding_dim", 128)

    mlflow.log_param("train_size", len(train_subset))
    mlflow.log_param("valid_size", len(valid_subset))
    mlflow.log_param("test_size", len(test_dataset))

    mlflow.log_param("attention_heads", 8)
    mlflow.log_param("dropout_rate", 0.1)
    mlflow.log_param("cnn_filters", "32-64-96")
    mlflow.log_param("drug_kernels", "4-6-8")
    mlflow.log_param("protein_kernels", "4-6-12")

    mlflow.log_param("loss_function", "MSELoss")
    mlflow.log_param("device", str(device))
    mlflow.log_param("seed", seed)


    # --------------------------------------------------------
    # Early-Stopping-Variablen
    # --------------------------------------------------------

    # Anzahl der Epochen ohne Verbesserung, bevor das Training abbricht
    patience = 50

    # Bester bisher beobachteter Validation-MSE
    best_valid_mse = float("inf")

    # Speichert den Modellzustand mit dem besten Validation-MSE
    best_model_state = None

    # Zähler für Epochen ohne Verbesserung
    epochs_without_improvement = 0

    mlflow.log_param("early_stopping_patience", patience)
    mlflow.log_param("early_stopping_metric", "valid_mse")


    # --------------------------------------------------------
    # Trainingsschleife
    # --------------------------------------------------------

    for epoch in range(epochs):

        # Modell in Trainingsmodus setzen
        model.train()

        # Liste zur Speicherung der Batch-Losses einer Epoche
        train_losses = []

        # -----------------------------
        # Training über alle Mini-Batches
        # -----------------------------
        for smiles_batch, protein_batch, y_batch in train_loader:

            # Batch-Daten auf GPU/CPU verschieben
            smiles_batch = smiles_batch.to(device)
            protein_batch = protein_batch.to(device)
            y_batch = y_batch.to(device)

            # Gradienten aus vorherigem Schritt zurücksetzen
            optimizer.zero_grad()

            # Vorhersage berechnen
            predictions = model(smiles_batch, protein_batch)

            # Fehler zwischen Vorhersage und Zielwert berechnen
            loss = loss_fn(predictions, y_batch)

            # Backpropagation
            loss.backward()

            # Modellparameter aktualisieren
            optimizer.step()

            # Lernrate gemäß Scheduler aktualisieren
            scheduler.step()

            # Batch-Loss speichern
            train_losses.append(loss.item())

        # Durchschnittlicher Trainings-MSE der Epoche
        train_mse = sum(train_losses) / len(train_losses)


        # ----------------------------------------------------
        # Validierung nach jeder Epoche
        # ----------------------------------------------------

        # Modell in Evaluationsmodus setzen
        model.eval()

        valid_preds = []
        valid_true = []
        valid_losses = []

        # Während der Validierung werden keine Gradienten benötigt
        with torch.no_grad():
            for smiles_batch, protein_batch, y_batch in valid_loader:

                # Validierungsdaten auf GPU/CPU verschieben
                smiles_batch = smiles_batch.to(device)
                protein_batch = protein_batch.to(device)
                y_batch = y_batch.to(device)

                # Vorhersage auf Validierungsdaten
                preds = model(smiles_batch, protein_batch)

                # Validierungs-Loss berechnen
                valid_loss = loss_fn(preds, y_batch)

                # Loss, Vorhersagen und Zielwerte speichern
                valid_losses.append(valid_loss.item())
                valid_preds.extend(preds.cpu().numpy().flatten())
                valid_true.extend(y_batch.cpu().numpy().flatten())

        # Validierungsmetriken berechnen
        valid_loss_mean = sum(valid_losses) / len(valid_losses)
        valid_mse = mean_squared_error(valid_true, valid_preds)
        valid_mae = mean_absolute_error(valid_true, valid_preds)
        valid_r2 = r2_score(valid_true, valid_preds)

        # Fortschritt in der Konsole ausgeben
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train MSE: {train_mse:.4f} | "
            f"Valid MSE: {valid_mse:.4f} | "
            f"Valid MAE: {valid_mae:.4f} | "
            f"Valid R2: {valid_r2:.4f}"
        )

        # Trainings- und Validierungsmetriken in MLflow loggen
        mlflow.log_metric("train_mse", train_mse, step=epoch + 1)
        mlflow.log_metric("valid_loss", valid_loss_mean, step=epoch + 1)
        mlflow.log_metric("valid_mse", valid_mse, step=epoch + 1)
        mlflow.log_metric("valid_mae", valid_mae, step=epoch + 1)
        mlflow.log_metric("valid_r2", valid_r2, step=epoch + 1)

        # Aktuelle Lernrate loggen
        current_lr = optimizer.param_groups[0]["lr"]
        mlflow.log_metric("learning_rate_current", current_lr, step=epoch + 1)


        # ----------------------------------------------------
        # Early Stopping auf Basis des Validation-MSE
        # ----------------------------------------------------

        # Falls sich der Validation-MSE verbessert, wird das Modell gespeichert
        if valid_mse < best_valid_mse:
            best_valid_mse = valid_mse
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            mlflow.log_metric("best_valid_mse", best_valid_mse, step=epoch + 1)

        # Falls keine Verbesserung erreicht wurde, wird der Zähler erhöht
        else:
            epochs_without_improvement += 1

        # Training abbrechen, wenn über patience Epochen keine Verbesserung erfolgt
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping after epoch {epoch + 1}. "
                f"Best valid MSE: {best_valid_mse:.4f}"
            )
            mlflow.log_param("stopped_epoch", epoch + 1)
            break


    # --------------------------------------------------------
    # Bestes Modell laden
    # --------------------------------------------------------

    # Finalen besten Validation-MSE und tatsächlich trainierte Epochen loggen
    mlflow.log_metric("best_valid_mse_final", best_valid_mse)
    mlflow.log_param("epochs_trained", epoch + 1)

    # Modellzustand mit bestem Validation-MSE wiederherstellen
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    # --------------------------------------------------------
    # Finale Evaluation auf unabhängigem Testset
    # --------------------------------------------------------

    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for smiles_batch, protein_batch, y_batch in test_loader:

            # Testdaten auf GPU/CPU verschieben
            smiles_batch = smiles_batch.to(device)
            protein_batch = protein_batch.to(device)

            # Vorhersage berechnen
            preds = model(smiles_batch, protein_batch)

            # Vorhersagen und Zielwerte sammeln
            all_preds.extend(preds.cpu().numpy().flatten())
            all_true.extend(y_batch.cpu().numpy().flatten())

    # Testmetriken berechnen
    mse = mean_squared_error(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)

    # Testmetriken in der Konsole ausgeben
    print("Test MSE:", mse)
    print("Test MAE:", mae)
    print("Test R2:", r2)

    # Testmetriken in MLflow loggen
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)

    # Trainiertes Modell als MLflow-Artefakt speichern
    mlflow.pytorch.log_model(model, name="model")
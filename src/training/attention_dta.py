import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import mlflow.pytorch
from src.models.AttentionDTA import AttentionDTA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

learning_rate = 0.0001
batch_size = 128
epochs = 20

train_data = torch.load("data/encoded/encoded_train.pt")
test_data = torch.load("data/encoded/encoded_test.pt")

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

# Create dataloaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AttentionDTA().to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

mlruns_path = Path("mlruns").resolve()
run_name = "AttentionDTA_Baseline"
mlflow.set_tracking_uri(mlruns_path.as_uri())
mlflow.set_experiment("dta_model_comparison")

with mlflow.start_run(run_name=run_name):

    mlflow.log_param("model", "AttentionDTA")
    mlflow.log_param("epochs", epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("smiles_max_len", 100)
    mlflow.log_param("protein_max_len", 1200)
    mlflow.log_param("embedding_dim", 128)
    mlflow.log_param("train_size", len(train_dataset))
    mlflow.log_param("test_size", len(test_dataset))
    mlflow.log_param("attention_heads", 8)
    mlflow.log_param("dropout_rate", 0.1)
    mlflow.log_param("cnn_filters", "32-64-96")
    mlflow.log_param("drug_kernels", "4-6-8")
    mlflow.log_param("protein_kernels", "4-6-12")
    mlflow.log_param("loss_function", "MSELoss")
    mlflow.log_param("device", str(device))

    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for smiles_batch, protein_batch, y_batch in train_loader:
            smiles_batch = smiles_batch.to(device)
            protein_batch = protein_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(smiles_batch, protein_batch)
            loss = loss_fn(predictions, y_batch)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        train_mse = sum(train_losses)/len(train_losses)
        print(f"Epoch {epoch + 1}/{epochs} completed | Train MSE: {train_mse:.4f}")
        mlflow.log_metric("train_mse", train_mse, step=epoch + 1)


    # Evaluation
    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():
        for smiles_batch, protein_batch, y_batch in test_loader:
            smiles_batch = smiles_batch.to(device)
            protein_batch = protein_batch.to(device)
            preds = model(smiles_batch, protein_batch)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_true.extend(y_batch.numpy().flatten())

    mse = mean_squared_error(all_true, all_preds)
    mae = mean_absolute_error(all_true, all_preds)
    r2 = r2_score(all_true, all_preds)
   

    # Log test metrics to MLflow
    mlflow.log_metric("test_mse", mse)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_r2", r2)


    # Save model to MLflow
    mlflow.pytorch.log_model(model, "model")
import torch
import torch.nn as nn



# Define the model architecture
class AttentionDTA(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.smiles_embedding = nn.Embedding(65, 128, padding_idx=0)
        self.protein_embedding = nn.Embedding(26, 128, padding_idx=0)
        
        self.smiles_cnn = nn.Conv1d(128, 96, kernel_size=8)
        self.protein_cnn = nn.Conv1d(128, 96, kernel_size=12)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(192, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, smiles, protein):
        smiles = self.smiles_embedding(smiles)
        protein = self.protein_embedding(protein)
        
        smiles = smiles.permute(0, 2, 1)
        protein = protein.permute(0, 2, 1)
        
        smiles = torch.relu(self.smiles_cnn(smiles))
        protein = torch.relu(self.protein_cnn(protein))
        
        smiles = self.pool(smiles).squeeze(-1)
        protein = self.pool(protein).squeeze(-1)
        
        combined = torch.cat([smiles, protein], dim=1)
        output = self.fc(combined)
        
        return output
    

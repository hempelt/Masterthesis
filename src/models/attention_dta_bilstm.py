import torch
import torch.nn as nn



class MultiHeadInteractionAttention(nn.Module):
    def __init__(self, feature_dim=96, num_heads=8):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.drug_projection = nn.Linear(feature_dim, feature_dim * num_heads)
        self.protein_projection = nn.Linear(feature_dim, feature_dim * num_heads)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.scale = feature_dim ** 0.5

    def forward(self, drug, protein):
        """
        drug:    [batch_size, 96, drug_seq_len]
        protein: [batch_size, 96, protein_seq_len]
        """

        batch_size, drug_features, drug_len = drug.shape
        batch_size, protein_features, protein_len = protein.shape

        # Original-nahe Attention-Projektion
        drug_att = self.relu(
            self.drug_projection(drug.permute(0, 2, 1))
        ).view(
            batch_size, self.num_heads, drug_len, self.feature_dim
        )

        protein_att = self.relu(
            self.protein_projection(protein.permute(0, 2, 1))
        ).view(
            batch_size, self.num_heads, protein_len, self.feature_dim
        )


        # Interaktionsmatrix zwischen Drug- und Proteinpositionen
        # Ergebnis: [batch, heads, drug_len, protein_len]
        interaction_map = torch.matmul(
            drug_att,
            protein_att.permute(0, 1, 3, 2)
        )

        interaction_map = self.tanh(interaction_map / self.scale)

        # Mittelwert über alle Heads
        # [batch, drug_len, protein_len]
        interaction_map = torch.mean(interaction_map, dim=1)

        # Drug-Attention: Wichtigkeit jeder Drug-Position
        # Summe über Proteinpositionen
        drug_attention = self.tanh(torch.sum(interaction_map, dim=2))

        # Protein-Attention: Wichtigkeit jeder Proteinposition
        # Summe über Drugpositionen
        protein_attention = self.tanh(torch.sum(interaction_map, dim=1))

        # Dimension anpassen für Multiplikation
        # [batch, seq_len] -> [batch, 1, seq_len]
        drug_attention = drug_attention.unsqueeze(1)
        protein_attention = protein_attention.unsqueeze(1)

        # CNN-Features gewichten
        drug_weighted = drug * drug_attention
        protein_weighted = protein * protein_attention

        return drug_weighted, protein_weighted





# Define the model architecture
class AttentionDTA(nn.Module):
    def __init__(self, smiles_vocab_size=65, protein_vocab_size=26):
        super().__init__()
        
        self.smiles_embedding = nn.Embedding(smiles_vocab_size, 128, padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size, 128, padding_idx=0)

        # BiLSTM für Protein-Kontextinformationen
        # hidden_size=64 und bidirectional=True ergibt wieder 128 Output-Dimensionen:
        # 64 forward + 64 backward = 128
        
        self.protein_bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # CNN block for SMILES string
        self.smiles_cnn = nn.Sequential(
        nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6),
        nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=96, kernel_size=8),
        nn.ReLU()
        )

        # CNN block for Protein sequence (in OriginalCode were protein_kernel = [4, 8, 12], but we use kernel size from article [4, 6, 12] to be comparable)
        self.protein_cnn = nn.Sequential(
        nn.Conv1d(in_channels=128, out_channels=32, kernel_size=4),
        nn.ReLU(),
        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6),
        nn.ReLU(),
        nn.Conv1d(in_channels=64, out_channels=96, kernel_size=12),
        nn.ReLU()
        )

        # Multi Head Attention-Block
        self.attention = MultiHeadInteractionAttention(
            feature_dim=96,
            num_heads=8
        )
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(192, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
            nn.LeakyReLU(),

            nn.Linear(512, 1)
        )
        
        # Output-Bias wie im Originalcode auf 5 setzen
        torch.nn.init.constant_(self.fc[-1].bias, 5)
    
    def forward(self, smiles, protein):
        # Integer-IDs -> Embedding-Vektoren
        smiles = self.smiles_embedding(smiles)
        protein = self.protein_embedding(protein)

        # BiLSTM für Protein-Kontextinformationen
        # Input:  [batch, protein_seq_len, 128]
        # Output: [batch, protein_seq_len, 128]
        protein, _ = self.protein_bilstm(protein)
        
        # Für Conv1d: [batch, seq_len, embedding_dim] -> [batch, embedding_dim, seq_len]
        smiles = smiles.permute(0, 2, 1)
        protein = protein.permute(0, 2, 1)
        
        # CNN Feature Extraction
        smiles = self.smiles_cnn(smiles)
        protein = self.protein_cnn(protein)

        # Attention zwischen SMILES- und Protein-CNN-Features
        smiles, protein = self.attention(smiles, protein)
        
        # Global Max Pooling
        smiles = self.pool(smiles).squeeze(-1)
        protein = self.pool(protein).squeeze(-1)
        
        # Features kombinieren
        combined = torch.cat([smiles, protein], dim=1)

        # MLP / Regression Head
        output = self.fc(combined)
        
        return output
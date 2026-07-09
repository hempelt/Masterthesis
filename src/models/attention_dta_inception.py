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


class Inception1DBlock(nn.Module):
    def __init__(self, in_channels=128, branch_channels=32, kernel_sizes=(4, 8, 16)):
        super().__init__()

        self.branch_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=kernel_sizes[0]
            ),
            nn.ReLU()
        )

        self.branch_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=kernel_sizes[1]
            ),
            nn.ReLU()
        )

        self.branch_3 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=kernel_sizes[2]
            ),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: [batch_size, in_channels, seq_len]

        Drei parallele CNNs mit unterschiedlichen Kernelgrößen.
        Jeder Branch erzeugt branch_channels Feature Maps.
        Danach werden die Feature Maps konkateniert.
        """

        x1 = self.branch_1(x)
        x2 = self.branch_2(x)
        x3 = self.branch_3(x)

        # Wegen unterschiedlicher Kernelgrößen entstehen unterschiedliche Sequenzlängen.
        # Für die Konkatenation werden alle Outputs auf die kleinste Länge gekürzt.
        min_len = min(x1.size(-1), x2.size(-1), x3.size(-1))

        x1 = x1[:, :, :min_len]
        x2 = x2[:, :, :min_len]
        x3 = x3[:, :, :min_len]

        # Ergebnis: [batch_size, branch_channels * 3, min_len]
        # Bei branch_channels=32 also: [batch_size, 96, min_len]
        out = torch.cat([x1, x2, x3], dim=1)

        return out


class AttentionDTA(nn.Module):
    def __init__(self, smiles_vocab_size=65, protein_vocab_size=16):
        super().__init__()

        self.smiles_embedding = nn.Embedding(smiles_vocab_size, 128,padding_idx=0)
        self.protein_embedding = nn.Embedding(protein_vocab_size,128,padding_idx=0)

        # Inception-Block für SMILES
        # Drei parallele CNNs mit Kernelgrößen 4, 8, 16
        # 32 + 32 + 32 = 96 Output-Kanäle
        self.smiles_cnn = Inception1DBlock(
            in_channels=128,
            branch_channels=32,
            kernel_sizes=(4, 8, 16)
        )

        # Inception-Block für Proteinsequenz
        # Ebenfalls 96 Output-Kanäle, damit der Attention-Block unverändert bleibt
        self.protein_cnn = Inception1DBlock(
            in_channels=128,
            branch_channels=32,
            kernel_sizes=(4, 8, 16)
        )

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

        torch.nn.init.constant_(self.fc[-1].bias, 5)

    def forward(self, smiles, protein):
        # Integer-IDs -> Embedding-Vektoren
        smiles = self.smiles_embedding(smiles)
        protein = self.protein_embedding(protein)

        # Für Conv1d:
        # [batch, seq_len, embedding_dim] -> [batch, embedding_dim, seq_len]
        smiles = smiles.permute(0, 2, 1)
        protein = protein.permute(0, 2, 1)

        # Inception Feature Extraction
        smiles = self.smiles_cnn(smiles)
        protein = self.protein_cnn(protein)

        # Attention zwischen SMILES- und Protein-Features
        smiles, protein = self.attention(smiles, protein)

        # Global Max Pooling
        smiles = self.pool(smiles).squeeze(-1)
        protein = self.pool(protein).squeeze(-1)

        # Features kombinieren
        combined = torch.cat([smiles, protein], dim=1)

        # MLP / Regression Head
        output = self.fc(combined)

        return output
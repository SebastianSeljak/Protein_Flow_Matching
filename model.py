import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class FoldingFlow(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, num_layers=6, dim_feedforward=1024):
        super().__init__()
        self.aa_embedding = nn.Embedding(21, embed_dim) # 20 AAs + 1 for padding
        self.pos_encoding = nn.Parameter(torch.randn(1, 8192, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.output_head = nn.Sequential(
            nn.Linear(embed_dim + embed_dim + 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # velocity for phi and psi
        )

    def forward(self, t, x_t, seq, embeddings=None):
        """
        Args:
            t: (B,) scalar time
            x_t: (B, L_torsion, 2) noisy phi/psi
            seq: (B, L_seq) amino acid indices
            embeddings: (B, L_seq, d_esm) optional pre-computed embeddings
        Returns:
            v_t: (B, L_torsion, 2) predicted velocity
        """
        B, L_torsion, _ = x_t.shape
        B, L_seq = seq.shape
        
        # 1. Sequence feature from Transformer
        if embeddings is not None:
            # If embeddings are provided (e.g. ESM-2 320-dim), use them
            # We assume they are already projected or we project them here if needed
            if embeddings.shape[-1] != self.aa_embedding.embedding_dim:
                if not hasattr(self, 'embedding_projection'):
                    self.embedding_projection = nn.Linear(embeddings.shape[-1], self.aa_embedding.embedding_dim).to(embeddings.device)
                seq_emb = self.embedding_projection(embeddings)
            else:
                seq_emb = embeddings
        else:
            seq_emb = self.aa_embedding(seq) # (B, L_seq, embed_dim)
            
        # Ensure positional encoding is large enough or sliced correctly
        seq_emb = seq_emb + self.pos_encoding[:, :L_seq, :]
        
        mask = (seq == 0) # Assuming 0 is pad
        seq_feats = self.transformer(seq_emb, src_key_padding_mask=mask) # (B, L_seq, embed_dim)
        
        # 2. Align sequence features with torsions if necessary
        # Torsion angles (phi, psi) are usually missing for the very first/last residues
        if L_seq != L_torsion:
            diff = L_seq - L_torsion
            start = diff // 2
            seq_feats = seq_feats[:, start : start + L_torsion, :]
        
        # 3. Time embedding
        t_emb = self.time_mlp(t) # (B, embed_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, L_torsion, -1) # (B, L_torsion, embed_dim)
        
        # 4. Concatenate and predict
        combined = torch.cat([seq_feats, t_emb, x_t], dim=-1) # (B, L_torsion, 2*embed_dim + 2)
        v_t = self.output_head(combined)
        
        return v_t

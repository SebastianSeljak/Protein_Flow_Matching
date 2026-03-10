import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Import local modules
from geometry import torsion_to_cartesian
from model import FoldingFlow

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PICKLE_PATH = 'angles_by_acid.pkl'
MODEL_SAVE_PATH = "model_stage3_normalized.pth"

# Model Architecture
EMBED_DIM = 256
NHEAD = 8
NUM_LAYERS = 6

# Training Params
BATCH_LIMIT = 500 
NUM_EPOCHS = 50    
LEARNING_RATE = 1e-4
MAX_LEN = 512

# AA Mapping
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_TO_IDX["PAD"] = 0

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
def load_data():
    if not os.path.exists(PICKLE_PATH):
        raise FileNotFoundError(f"Missing {PICKLE_PATH}. Please ensure data is available.")

    print(f"Loading data from {PICKLE_PATH}...")
    with open(PICKLE_PATH, 'rb') as f:
        df = pickle.load(f)
    
    # Identify unique proteins
    df['protein_id'] = (df['pos'] == 1).cumsum()

    all_data = []
    for pid, group in tqdm(df.groupby('protein_id', sort=False), desc="Processing data"):
        full_seq = group['sequence'].iloc[0]
        if len(full_seq) > MAX_LEN: continue 
        
        indices = group['pos'].values - 1
        seq_indices = [AA_TO_IDX.get(full_seq[idx], 0) if idx < len(full_seq) else 0 for idx in indices]
        
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long)
        
        # Convert degrees to radians for scale alignment with noise
        angle_tensor = torch.tensor(group[['x', 'y']].values, dtype=torch.float32) * (np.pi / 180.0)
        
        if len(seq_tensor) == len(angle_tensor):
            all_data.append({'seq': seq_tensor, 'angles': angle_tensor, 'id': pid})
            
    return all_data

# ==========================================
# 3. Loss Functions
# ==========================================
def wrapped_torsion_loss(pred_v, target_v):
    """Wrapped loss in Radians"""
    diff = pred_v - target_v
    return torch.mean(1 - torch.cos(diff))

def structural_loss(pred_angles, target_angles):
    """Geometric loss (converting radians back to degrees for NeRF engine)"""
    if torch.isnan(pred_angles).any(): 
        return torch.tensor(0.0, device=pred_angles.device)
        
    # Convert radians to degrees for geometry.py
    pred_coords = torsion_to_cartesian(
        torch.rad2deg(pred_angles[:, 0].unsqueeze(0)), 
        torch.rad2deg(pred_angles[:, 1].unsqueeze(0))
    )
    target_coords = torsion_to_cartesian(
        torch.rad2deg(target_angles[:, 0].unsqueeze(0)), 
        torch.rad2deg(target_angles[:, 1].unsqueeze(0))
    )
    
    # Standardize coordinates
    pred_coords = pred_coords - pred_coords.mean(dim=1, keepdim=True)
    target_coords = target_coords - target_coords.mean(dim=1, keepdim=True)
    return F.mse_loss(pred_coords, target_coords)

# ==========================================
# 4. Training Pipeline
# ==========================================
def main():
    print(f"Using device: {DEVICE}")

    # Load and Split Data
    all_data = load_data()
    np.random.shuffle(all_data)
    split = int(0.95 * len(all_data))
    train_data = all_data[:split]
    
    # Initialize Model
    model = FoldingFlow(embed_dim=EMBED_DIM, nhead=NHEAD, num_layers=NUM_LAYERS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"Starting Stage 3 Training for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        # Staged Cartesian weight (ramps up over time)
        cartesian_weight = min(0.05, epoch * 0.002) if epoch > 5 else 0.0
        
        # Batch processing (using limit since we have single-sequence training style)
        indices = np.random.choice(len(train_data), min(len(train_data), BATCH_LIMIT), replace=False)
        
        pbar = tqdm([train_data[i] for i in indices], desc=f"Epoch {epoch}")
        for item in pbar:
            seq = item['seq'].unsqueeze(0).to(DEVICE)
            x1 = item['angles'].unsqueeze(0).to(DEVICE)
            
            # Flow matching time step
            t = torch.rand(1).to(DEVICE)
            x0 = torch.randn_like(x1) # Pure noise
            
            # Linear interpolation (Probability Flow ODE)
            xt = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            
            # Model prediction
            pred_v = model(t, xt, seq)
            
            # Torsion Loss
            loss_torsion = wrapped_torsion_loss(pred_v, target_v)
            
            loss = loss_torsion
            
            # (Optional) Structural Loss for refinement
            if cartesian_weight > 0:
                x1_pred = xt + (1 - t) * pred_v
                loss_cartesian = structural_loss(x1_pred.squeeze(0), x1.squeeze(0))
                loss += cartesian_weight * loss_cartesian
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        avg_loss = epoch_loss / len(indices)
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f} (CW: {cartesian_weight:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            print(f"Saved checkpoint to checkpoint_epoch_{epoch}.pth")

    # Final Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Final model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

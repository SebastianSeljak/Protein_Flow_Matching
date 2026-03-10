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
from geometry import torsion_to_cartesian, kabsch_alignment
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
NUM_EPOCHS = 40    
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

    # Aggressive mocking to handle version mismatches in pickled transformers models
    import sys
    import types
    import transformers
    
    def setup_mocks():
        # Mock core_model_loading if missing
        if 'transformers.core_model_loading' not in sys.modules:
            m = types.ModuleType('transformers.core_model_loading')
            sys.modules['transformers.core_model_loading'] = m
            class MockWeightRenaming:
                def __init__(self, *args, **kwargs): pass
                def __call__(self, *args, **kwargs): return args[0]
            m.WeightRenaming = MockWeightRenaming
        
        # Mock tokenization_python if missing
        if 'transformers.tokenization_python' not in sys.modules:
            m = types.ModuleType('transformers.tokenization_python')
            sys.modules['transformers.tokenization_python'] = m
            # Redirect common tokenizer base classes/utils if needed
            from transformers.models.esm import EsmTokenizer
            m.EsmTokenizer = EsmTokenizer
            from transformers import PreTrainedTokenizer
            m.PreTrainedTokenizer = PreTrainedTokenizer
            
            # Use the real Trie from the current version of transformers
            try:
                from transformers.tokenization_utils import Trie
                m.Trie = Trie
            except ImportError:
                class MockTrie:
                    def __init__(self, *args, **kwargs): pass
                    def add(self, *args, **kwargs): pass
                    def get(self, *args, **kwargs): return None
                    def split(self, text): return [text]
                m.Trie = MockTrie

    setup_mocks()

    # Check for ESM-2 model to generate embeddings
    esm_model_path = 'models/esm2_8M.pkl'
    esm_tokenizer_path = 'models/esm2_8M_tokenizer.pkl'
    
    esm_model = None
    tokenizer = None
    
    if os.path.exists(esm_model_path) and os.path.exists(esm_tokenizer_path):
        print(f"Loading local ESM-2 from {esm_model_path}...")
        try:
            with open(esm_tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            with open(esm_model_path, 'rb') as f:
                esm_model = pickle.load(f)
            print("Successfully loaded local ESM-2.")
        except Exception as e:
            print(f"Error loading local ESM-2 pickles: {e}")
            raise e # User requested no fallback, so we fail if it fails

    if esm_model is not None:
        esm_model.to(DEVICE)
        esm_model.eval()

    all_data = []
    for pid, group in tqdm(df.groupby('protein_id', sort=False), desc="Processing data"):
        full_seq = group['sequence'].iloc[0]
        if len(full_seq) > MAX_LEN: continue 
        
        indices = group['pos'].values - 1
        seq_indices = [AA_TO_IDX.get(full_seq[idx], 0) if idx < len(full_seq) else 0 for idx in indices]
        
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long)
        
        # Convert degrees to radians for scale alignment with noise
        angle_tensor = torch.tensor(group[['x', 'y']].values, dtype=torch.float32) * (np.pi / 180.0)
        
        # Generate ESM-2 embeddings if model is available
        emb_tensor = None
        if esm_model is not None and tokenizer is not None:
            with torch.no_grad():
                inputs = tokenizer(full_seq, return_tensors='pt').to(DEVICE)
                outputs = esm_model(**inputs)
                # last_hidden_state shape: (1, L+2, 320)
                # We need states for specific positions (pos - 1 + 1 for CLS)
                hidden_states = outputs.last_hidden_state[0] # (L+2, 320)
                emb_indices = indices + 1 # +1 for CLS token
                emb_tensor = hidden_states[emb_indices].cpu() # (N_res, 320)

        if len(seq_tensor) == len(angle_tensor):
            data_item = {'seq': seq_tensor, 'angles': angle_tensor, 'id': pid}
            if emb_tensor is not None:
                data_item['embeddings'] = emb_tensor
            all_data.append(data_item)
            
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
        return torch.tensor(0.0, device=pred_angles.device), 0.0
        
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
    
    # Apply Kabsch alignment to align pred_coords to target_coords
    aligned_pred = kabsch_alignment(pred_coords, target_coords)
    
    mse = F.mse_loss(aligned_pred, target_coords)
    rmse = torch.sqrt(mse).item()
    
    return mse, rmse

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
        epoch_rmse = 0
        rmse_counts = 0
        
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
            emb = item.get('embeddings')
            if emb is not None:
                emb = emb.unsqueeze(0).to(DEVICE)
                
            pred_v = model(t, xt, seq, embeddings=emb)
            
            # Torsion Loss
            loss_torsion = wrapped_torsion_loss(pred_v, target_v)
            
            loss = loss_torsion
            current_rmse = 0.0
            
            # (Optional) Structural Loss for refinement
            if cartesian_weight > 0:
                x1_pred = xt + (1 - t) * pred_v
                loss_cartesian, current_rmse = structural_loss(x1_pred.squeeze(0), x1.squeeze(0))
                loss += cartesian_weight * loss_cartesian
                epoch_rmse += current_rmse
                rmse_counts += 1
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            postfix = {'loss': f"{loss.item():.4f}"}
            if cartesian_weight > 0:
                postfix['rmse'] = f"{current_rmse:.4f}"
            pbar.set_postfix(postfix)
            
        scheduler.step()
        avg_loss = epoch_loss / len(indices)
        avg_rmse = epoch_rmse / rmse_counts if rmse_counts > 0 else 0.0
        print(f"Epoch {epoch} Avg Loss: {avg_loss:.4f} | Avg RMSE: {avg_rmse:.4f} (CW: {cartesian_weight:.4f})")

    # Final Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training complete. Final model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()

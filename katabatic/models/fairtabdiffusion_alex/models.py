import torch
import torch.optim as optim
import numpy as np
import os
from tqdm.auto import tqdm
from .modules import DenoiseFn, GaussianMultinomialDiffusion

class FairTabDiffusion:
    def __init__(self, num_classes, input_dim, device='cuda', lr=2e-4, weight_decay=1e-4):
        self.device = device
        self.num_classes = num_classes # List of cardinalities for each column
        
        # Calculate input dim (sum of one-hot lengths)
        self.input_dim_ohe = sum(num_classes)
        
        self.denoise_fn = DenoiseFn(
            input_dim=self.input_dim_ohe,
            d_t_emb=128, 
            d_cond_emb=128, 
            n_channels=64
        ).to(device)
        
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=np.array(num_classes),
            num_numerical_features=0, # Treating all as categorical for stability
            denoise_fn=self.denoise_fn,
            device=device
        ).to(device)
        
        self.optimizer = optim.AdamW(
            self.diffusion.parameters(), lr=lr, weight_decay=weight_decay
        )

    def train(self, data_loader, epochs=100):
        self.diffusion.train()
        with tqdm(range(epochs), desc="Training FairTabDiffusion") as pbar:
            for _ in pbar:
                batch_losses = []
                for x, y in data_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    self.optimizer.zero_grad()
                    loss = self.diffusion.mixed_loss(x, y)
                    loss.backward()
                    self.optimizer.step()
                    
                    batch_losses.append(loss.item())
                
                # Check for NaN loss
                avg_loss = np.mean(batch_losses)
                if np.isnan(avg_loss):
                    print("Warning: Loss is NaN. Training might be unstable.")
                
                pbar.set_postfix({'loss': avg_loss})

    # Renamed from generate -> sample to match common APIs
    def sample(self, n_samples, cond_labels):
        self.diffusion.eval()
        cond = torch.tensor(cond_labels).to(self.device)
        
        # Generate
        samples = self.diffusion.sample(n_samples, cond)
        return samples.cpu().numpy()

    # --- Compliance Methods (in case of inheritance) ---
    def evaluate(self, X, y=None, **kwargs):
        return {} # Placeholder

    def save(self, path):
        torch.save(self.diffusion.state_dict(), path)

    def load(self, path):
        self.diffusion.load_state_dict(torch.load(path))
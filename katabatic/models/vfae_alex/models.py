import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm
from .modules import VariationalFairAutoEncoder
from .utils import loss_function_vfae

class VFAE:
    def __init__(self, x_dim, s_dim, y_dim, z_dim=50, device='cpu'):
        self.device = device
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.s_dim = s_dim
        self.y_dim = y_dim
        
        self.model = VariationalFairAutoEncoder(
            x_dim=x_dim,
            s_dim=s_dim,
            y_dim=y_dim,
            z1_enc_dim=100,
            z2_enc_dim=100,
            z1_dec_dim=100,
            x_dec_dim=100,
            z_dim=z_dim,
            dropout_rate=0.1
        ).to(device)
        
        # Reduced Learning Rate for stability (1e-3 -> 5e-4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)

    def train(self, x_data, s_data, y_data, epochs=100, batch_size=64):
        self.model.train()
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(x_data), 
            torch.FloatTensor(s_data), 
            torch.FloatTensor(y_data)
        )
        # Drop last batch to avoid unstable small batches
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        with tqdm(range(epochs), desc="Training VFAE") as pbar:
            for _ in pbar:
                epoch_loss = 0
                valid_batches = 0
                
                for x_b, s_b, y_b in loader:
                    x_b, s_b, y_b = x_b.to(self.device), s_b.to(self.device), y_b.to(self.device)
                    
                    inputs = {'x': x_b, 's': s_b, 'y': y_b}
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    loss = loss_function_vfae(outputs, inputs)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        # Skip this batch but continue training
                        continue
                        
                    loss.backward()
                    
                    # --- STABILITY FIX: Gradient Clipping ---
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                    
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    valid_batches += 1
                
                # Avoid division by zero in logging
                avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
                pbar.set_postfix({'loss': f"{avg_loss:.4f}"})

    def generate(self, n_samples, s_marginal, y_marginal):
        self.model.eval()
        with torch.no_grad():
            # 1. Sample Z2
            z2 = torch.randn(n_samples, self.z_dim).to(self.device)
            
            # 2. Sample Y
            y_idxs = np.random.choice(len(y_marginal), n_samples, replace=True)
            y_sample = torch.FloatTensor(y_marginal[y_idxs]).to(self.device)
            
            # 3. Decode Z2 -> Z1
            z2_y = torch.cat([z2, y_sample], dim=1)
            # The decoder returns (z, sigma, mu), we usually take the mean (mu) for clean generation
            # or sample again. Taking 'z' (sampled) is fine too.
            z1_gen, _, _ = self.model.decoder_z1(z2_y)
            
            # 4. Sample S
            s_idxs = np.random.choice(len(s_marginal), n_samples, replace=True)
            s_sample = torch.FloatTensor(s_marginal[s_idxs]).to(self.device)
            
            # 5. Decode Z1 -> X
            z1_s = torch.cat([z1_gen, s_sample], dim=1)
            x_recon = self.model.decoder_x(z1_s)
            
            x_generated = x_recon[:, :self.x_dim]
            
            return x_generated.cpu().numpy(), s_sample.cpu().numpy(), y_sample.cpu().numpy()
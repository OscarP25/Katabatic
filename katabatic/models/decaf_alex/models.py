import torch
import torch.optim as optim
import networkx as nx
import numpy as np
from tqdm.auto import tqdm
from .modules import Generator_causal, Discriminator, trace_expm

class DECAF:
    def __init__(
        self,
        input_dim: int,
        dag_seed: list = [],
        h_dim: int = 200,
        lr: float = 1e-3,
        batch_size: int = 64,
        lambda_gp: float = 10,
        lambda_privacy: float = 1,
        rho: float = 1,
        alpha: float = 1,
        l1_g: float = 0,
        l1_W: float = 1,
        eps: float = 1e-8,
        device: str = 'cpu'
    ):
        self.x_dim = input_dim
        self.z_dim = input_dim # In DECAF z_dim = x_dim
        self.device = device
        self.dag_seed = dag_seed
        
        # Hyperparams
        self.lr = lr
        self.batch_size = batch_size
        self.lambda_gp = lambda_gp
        self.lambda_privacy = lambda_privacy
        self.rho = rho
        self.alpha = alpha
        self.l1_g = l1_g
        self.l1_W = l1_W
        self.eps = eps

        # Networks
        self.generator = Generator_causal(
            z_dim=self.z_dim, x_dim=self.x_dim, h_dim=h_dim, 
            device=device, dag_seed=dag_seed
        ).to(device)
        
        self.discriminator = Discriminator(
            x_dim=self.x_dim, h_dim=h_dim, device=device
        ).to(device)

        # Optimizers
        self.opt_g = optim.AdamW(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_d = optim.AdamW(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        alpha = torch.rand(real_samples.size(0), 1, device=self.device).expand(real_samples.size())
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        
        fake = torch.ones(real_samples.size(0), 1, device=self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def privacy_loss(self, real_samples, fake_samples):
        return -torch.mean(
            torch.sqrt(torch.mean((real_samples - fake_samples) ** 2, axis=1) + self.eps)
        )

    def l1_reg(self, model):
        l1 = torch.tensor(0.0, device=self.device)
        for name, layer in model.named_parameters():
            if "weight" in name:
                l1 = l1 + layer.norm(p=1)
        return l1

    def gradient_dag_loss(self, x, z):
        # Note: Expensive operation, usually omitted if DAG is provided
        x.requires_grad = True
        z.requires_grad = True
        gen_x = self.generator(x, z)
        dummy = torch.ones(x.size(0), device=self.device)
        W = torch.zeros(x.shape[1], x.shape[1], device=self.device)
        
        for i in range(x.shape[1]):
            gradients = torch.autograd.grad(
                outputs=gen_x[:, i], inputs=x, grad_outputs=dummy,
                create_graph=True, only_inputs=True
            )[0]
            W[i] = torch.sum(torch.abs(gradients), axis=0)
            
        h = trace_expm(W**2) - self.x_dim
        return 0.5 * self.rho * h * h + self.alpha * h

    def get_dag(self):
        return self.generator.M.detach().cpu().numpy()

    def get_gen_order(self):
        dense_dag = np.array(self.get_dag())
        dense_dag[dense_dag > 0.5] = 1
        dense_dag[dense_dag <= 0.5] = 0
        G = nx.from_numpy_array(dense_dag, create_using=nx.DiGraph)
        try:
            gen_order = list(nx.algorithms.dag.topological_sort(G))
        except nx.NetworkXUnfeasible:
            # Fallback if graph is not DAG (during training early stages)
            gen_order = list(range(self.x_dim))
        return gen_order

    def train(self, data, epochs=100):
        # Create DataLoader
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        self.generator.train()
        self.discriminator.train()

        with tqdm(range(epochs), desc="Training DECAF") as pbar:
            for _ in pbar:
                d_losses = []
                g_losses = []
                
                for batch in loader:
                    batch = batch[0].to(self.device)
                    
                    # --- 1. Train Discriminator ---
                    self.opt_d.zero_grad()
                    
                    z = torch.randn(batch.shape[0], self.z_dim, device=self.device)
                    fake = self.generator.sequential(batch, z, self.get_gen_order())
                    
                    real_loss = self.discriminator(batch).mean()
                    fake_loss = self.discriminator(fake.detach()).mean()
                    
                    d_loss = fake_loss - real_loss + self.lambda_gp * self.compute_gradient_penalty(batch, fake.detach())
                    
                    d_loss.backward()
                    self.opt_d.step()
                    d_losses.append(d_loss.item())
                    
                    # --- 2. Train Generator ---
                    self.opt_g.zero_grad()
                    
                    # Re-sample for generator update
                    z2 = torch.randn(batch.shape[0], self.z_dim, device=self.device)
                    fake = self.generator.sequential(batch, z2, self.get_gen_order())
                    
                    g_loss = -self.discriminator(fake).mean()
                    g_loss += self.lambda_privacy * self.privacy_loss(batch, fake)
                    g_loss += self.l1_g * self.l1_reg(self.generator)
                    
                    # If DAG is being learned (empty seed), add DAG loss
                    # Note: We skip complex gradient_dag_loss for speed unless requested
                    
                    g_loss.backward()
                    self.opt_g.step()
                    g_losses.append(g_loss.item())

                pbar.set_postfix({'d_loss': np.mean(d_losses), 'g_loss': np.mean(g_losses)})

    def generate(self, data, n_samples):
        self.generator.eval()
        with torch.no_grad():
            # DECAF generates by transforming input noise/data
            # Standard generation: Use random noise + sequential pass
            # We need a base 'x' to start masking from. 
            # In DECAF paper, 'x' input to sequential is dummy if full generation?
            # Actually, generator.sequential takes 'x'. 
            # If we want pure synthetic, we can pass zeros or sampled data?
            # DECAF usually works by imputation or transforming noise Z.
            
            # For sampling n new rows:
            z = torch.randn(n_samples, self.z_dim, device=self.device)
            
            # For the base 'x', we can sample from the real distribution (if performing counterfactuals)
            # OR initialize with zeros if the DAG handles full generation from Z.
            # Based on code: "out = x.clone().detach()" then "x_masked[:, i] = 0.0"
            # So the initial values of x don't matter for root nodes (parents), 
            # and child nodes get overwritten by parents. 
            # So passing zeros is safe.
            dummy_x = torch.zeros(n_samples, self.x_dim, device=self.device)
            
            generated = self.generator.sequential(dummy_x, z, self.get_gen_order())
            return generated.cpu().numpy()
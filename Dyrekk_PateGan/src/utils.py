import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        cat_dims: List[int],
        num_dims: int = 0,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.cat_dims = cat_dims
        self.hidden_dim = hidden_dim
        
        # Paper Appendix: Generator depth=3. 
        # Architecture: d -> d/2 -> d (where d is hidden_dim/feature dim)
        # We map latent -> d -> d/2 -> d
        self.layer1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )

        # Output heads for categorical columns
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_cat)
            for n_cat in cat_dims
        ])

        for head in self.heads:
            # Init to encourage uniform distribution at start
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

        self.num_dims = num_dims
        if num_dims > 0:
            self.num_head = nn.Linear(hidden_dim, num_dims)
        else:
            self.num_head = None
    
    def forward(self, z: torch.Tensor, temperature: float = 0.5, hard: bool = False) -> List[torch.Tensor]:
        x = self.layer1(z)
        x = self.layer2(x)
        h = self.layer3(x)
        
        outputs = []
        
        for head in self.heads:
            logits = head(h)
            # Gumbel-Softmax: differentiable sampling
            out = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)
            outputs.append(out)

        # Numeric outputs: one value per numeric column (continuous)
        if self.num_head is not None:
            num_out = self.num_head(h)
            # return each numeric column as a (batch,1) tensor in outputs
            for j in range(num_out.size(1)):
                outputs.append(num_out[:, j:j+1])
            
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator for categorical data.
    Accepts concatenated one-hot or soft-per-feature vectors.
    
    Paper Appendix:
    - Teacher (depth=1): "The number of hidden nodes in each layer is d".
      Structure: Input -> Hidden(d) -> Output(1)
    - Student (depth=3): "d -> d/2 -> d".
      Structure: Input -> Hidden(d) -> Hidden(d/2) -> Hidden(d) -> Output(1)
    """
    def __init__(
        self,
        cat_dims: List[int],
        hidden_dim: int = 128,
        depth: int = 3
    ):
        super().__init__()
        self.cat_dims = cat_dims
        self.depth = depth
        
        # Total input dimension (sum of all categorical dimensions)
        total_dim = sum(cat_dims)
        
        if depth == 1:
            # Teacher architecture: Input -> d -> 1
            # Note: Paper says "depth of teacher is 1" but also mentions "hidden nodes".
            # Usually depth 1 implies 1 hidden layer.
            self.net = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Student architecture: Input -> d -> d/2 -> d -> 1
            self.net = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
    
    def forward(self, x_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x_list: List of (batch, n_cat_i) one-hot or soft tensors
        Returns:
            (batch, 1) logits
        """
        x = torch.cat(x_list, dim=1)
        return self.net(x)
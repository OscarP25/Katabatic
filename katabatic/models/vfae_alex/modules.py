import torch
from torch.nn import Module, Linear, ReLU, Dropout

class VariationalFairAutoEncoder(Module):
    """
    Implementation of the Variational Fair AutoEncoder. 
    GitHub repository: https://github.com/yolomeus/dm-lab-2020-vfae/tree/master
    
    Paper: @misc{louizos2017variationalfairautoencoder,
      title={The Variational Fair Autoencoder}, 
      author={Christos Louizos and Kevin Swersky and Yujia Li and Max Welling and Richard Zemel},
      year={2017},
      eprint={1511.00830},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1511.00830}, 
}
    """
    def __init__(self,
                 x_dim,
                 s_dim,
                 y_dim,
                 z1_enc_dim,
                 z2_enc_dim,
                 z1_dec_dim,
                 x_dec_dim,
                 z_dim,
                 dropout_rate,
                 activation=ReLU()):
        super().__init__()
        y_out_dim = 2 if y_dim == 1 else y_dim

        self.encoder_z1 = VariationalMLP(x_dim + s_dim, z1_enc_dim, z_dim, activation)
        self.encoder_z2 = VariationalMLP(z_dim + y_dim, z2_enc_dim, z_dim, activation)

        self.decoder_z1 = VariationalMLP(z_dim + y_dim, z1_dec_dim, z_dim, activation)
        self.decoder_y = DecoderMLP(z_dim, x_dec_dim, y_out_dim, activation)
        self.decoder_x = DecoderMLP(z_dim + s_dim, x_dec_dim, x_dim + s_dim, activation)

        self.dropout = Dropout(dropout_rate)

    def forward(self, inputs):
        x, s, y = inputs['x'], inputs['s'], inputs['y']
        
        # Encode Z1
        x_s = torch.cat([x, s], dim=1)
        x_s = self.dropout(x_s)
        z1_encoded, z1_enc_logvar, z1_enc_mu = self.encoder_z1(x_s)

        # Encode Z2
        z1_y = torch.cat([z1_encoded, y], dim=1)
        z2_encoded, z2_enc_logvar, z2_enc_mu = self.encoder_z2(z1_y)

        # Decode Z1
        z2_y = torch.cat([z2_encoded, y], dim=1)
        z1_decoded, z1_dec_logvar, z1_dec_mu = self.decoder_z1(z2_y)

        # Decode X
        z1_s = torch.cat([z1_decoded, s], dim=1)
        x_decoded = self.decoder_x(z1_s)

        # Decode Y
        y_decoded = self.decoder_y(z1_encoded)

        outputs = {
            'x_decoded': x_decoded,
            'y_decoded': y_decoded,
            'z1_encoded': z1_encoded,
            'z1_enc_logvar': z1_enc_logvar,
            'z1_enc_mu': z1_enc_mu,
            'z2_enc_logvar': z2_enc_logvar,
            'z2_enc_mu': z2_enc_mu,
            'z1_dec_logvar': z1_dec_logvar,
            'z1_dec_mu': z1_dec_mu
        }

        return outputs

class VariationalMLP(Module):
    def __init__(self, in_features, hidden_dim, z_dim, activation):
        super().__init__()
        self.encoder = Linear(in_features, hidden_dim)
        self.activation = activation

        self.logvar_encoder = Linear(hidden_dim, z_dim)
        self.mu_encoder = Linear(hidden_dim, z_dim)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.activation(x)
        
        mu = self.mu_encoder(x)
        log_var_raw = self.logvar_encoder(x)
        
        # Implement clamping to avoid exploding/vanishing gradients
        log_var_clamped = torch.clamp(log_var_raw, min=-10, max=10)
        
        # Calculate std.dev
        sigma = (0.5 * log_var_clamped).exp()

        # compute reparameterisation
        epsilon = torch.randn_like(mu)
        z = mu + epsilon * sigma
        
        return z, sigma, mu

class DecoderMLP(Module):
    def __init__(self, in_features, hidden_dim, latent_dim, activation):
        super().__init__()
        self.lin_encoder = Linear(in_features, hidden_dim)
        self.activation = activation
        self.lin_out = Linear(hidden_dim, latent_dim)

    def forward(self, inputs):
        x = self.activation(self.lin_encoder(inputs))
        return self.lin_out(x)
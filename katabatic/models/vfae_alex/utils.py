import torch
import torch.nn.functional as F

def loss_function_vfae(outputs, inputs, alpha=1.0, beta=1.0):
    x_target = torch.cat([inputs['x'], inputs['s']], dim=1)
    y_target = inputs['y']
    
    # 1. Reconstruction Loss (Sum over batch)
    recon_loss = F.mse_loss(outputs['x_decoded'], x_target, reduction='sum')
    
    # 2. Supervised Loss
    if y_target.shape[1] > 1:
        # Cross Entropy handles LogSoftmax internally, usually stable
        sup_loss = F.cross_entropy(outputs['y_decoded'], torch.argmax(y_target, dim=1), reduction='sum')
    else:
        sup_loss = F.mse_loss(outputs['y_decoded'], y_target, reduction='sum')

    # 3. KL Divergence (Standard Normal Prior)
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Since we have sigma (not logvar), log(sigma^2) = 2 * log(sigma)
    def kl_term(mu, sigma):
        # FIX: Add epsilon to log calculation to prevent log(0) -> -inf
        sigma_safe = sigma + 1e-8
        
        # KL term calculation
        # sum over all dimensions
        element_kl = 1 + 2 * torch.log(sigma_safe) - mu.pow(2) - sigma.pow(2)
        return -0.5 * torch.sum(element_kl)

    kl_z1 = kl_term(outputs['z1_enc_mu'], outputs['z1_enc_logvar'])
    kl_z2 = kl_term(outputs['z2_enc_mu'], outputs['z2_enc_logvar'])

    return recon_loss + alpha * sup_loss + beta * (kl_z1 + kl_z2)
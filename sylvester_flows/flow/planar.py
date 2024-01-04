import torch
import torch.nn as nn
import torch.nn.functional as F

from core.abstract_flow import AbstractFlow
from core.planar import batch_compute_uhat, batch_sample_from_flow


class UnconditionalPlanarFlow(AbstractFlow):
    
    def __init__(self, D, K):
        
        super().__init__()

        self.D = D
        self.K = K
        
        assert self.D >= 1
        assert self.K >= 1
        
        self.μ = nn.Parameter(torch.zeros(1, self.D))
        self.pre_σ = nn.Parameter(torch.ones(1, self.D))

        self.u = nn.Parameter(torch.zeros(1, self.K, self.D))
        self.w = nn.Parameter(torch.randn(1, self.K, self.D))
        self.b = nn.Parameter(torch.randn(1, self.K, 1))

    @property
    def uhat(self):
        return batch_compute_uhat(self.u, self.w)
        
    @property
    def σ(self):
        return F.softplus(self.pre_σ)
    
    def sample_and_compute_logp(self, num_samples):
        samples, log_probs = batch_sample_from_flow(
            self.μ, self.σ, self.uhat, self.w, self.b, num_samples
        )    
        return samples[0], log_probs[0]
    
    def sample_from_each_layer(self, num_samples):
        samples_from_each_layer = batch_sample_from_flow(
            self.μ, self.σ, self.uhat, self.w, self.b, num_samples, return_z_from_each_layer=True
        )
        return samples_from_each_layer[0]

    
class ConditionalPlanarFlow(nn.Module):

    pass

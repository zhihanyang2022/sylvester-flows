import torch
import torch.nn as nn
import torch.nn.functional as F

from core.abstract_flow import AbstractFlow
from core.linalg import batch_construct_orthogonal
from core.sylvester import batch_sample_from_flow


class UnconditionalOrthogonalSylvesterFlow(AbstractFlow):
    
    def __init__(self, D, M, K):
        
        super().__init__()
    
        self.M = M
        self.D = D
        self.K = K
        
        assert self.D >= 1
        assert self.M >= 1
        assert self.D > self.M
        assert self.K >= 1
        
        self.μ = nn.Parameter(torch.zeros(1, self.D))
        self.pre_σ = nn.Parameter(torch.ones(1, self.D))
    
        self.pre_Q = nn.Parameter(torch.randn(1, self.K, self.D, self.M))
        
        # the flow is initialized to be an identity transformation (for easier training empirically)
        self.pre_R1 = nn.Parameter(torch.zeros(1, self.K, self.M, self.M))
        self.pre_R2 = nn.Parameter(torch.triu(torch.randn(1, self.K, self.M, self.M)))
        self.b = nn.Parameter(torch.randn(1, self.K, self.M))
        
        self.one_to_m = range(self.M)
        
    @property
    def σ(self):
        return F.softplus(self.pre_σ)
    
    @property
    def Q(self):
        return batch_construct_orthogonal(self.pre_Q)
    
    @property
    def R1(self):
        temp = torch.triu(self.pre_R1)
        temp[:, :, self.one_to_m, self.one_to_m] = torch.tanh(temp[:, :, self.one_to_m, self.one_to_m])
        return temp
    
    @property
    def R2(self):
        temp = torch.triu(self.pre_R2)
        temp[:, :, self.one_to_m, self.one_to_m] = torch.tanh(temp[:, :, self.one_to_m, self.one_to_m])
        return temp
    
    def sample_and_compute_logp(self, num_samples):
        samples, log_probs = batch_sample_from_flow(
            self.μ, self.σ, self.Q, self.R1, self.R2, self.b, num_samples
        )    
        return samples[0], log_probs[0]
    
    def sample_from_each_layer(self, num_samples):
        samples_from_each_layer = batch_sample_from_flow(
            self.μ, self.σ, self.Q, self.R1, self.R2, self.b, num_samples, return_z_from_each_layer=True
        )
        return samples_from_each_layer[0]

    
class ConditionalOrthogonalSylvesterFlow(nn.Module):

    pass

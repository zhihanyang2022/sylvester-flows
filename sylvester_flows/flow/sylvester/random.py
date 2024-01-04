import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ortho_group

import core


class UnconditionalRandomSylvesterFlow(nn.Module):
    
    def __init__(self, D, K):
        
        super().__init__()
    
        self.D = D
        self.K = K
        
        assert self.D >= 1
        assert self.K >= 1
        
        self.μ = nn.Parameter(torch.zeros(1, self.D))
        self.pre_σ = nn.Parameter(torch.ones(1, self.D))

        self.Q = torch.stack([
            torch.from_numpy(ortho_group.rvs(self.D)).float() for i in range(self.K)
        ]).unsqueeze(0)

        self.pre_R1 = nn.Parameter(torch.zeros(1, self.K, self.D, self.D))
        self.pre_R2 = nn.Parameter(torch.triu(torch.randn(1, self.K, self.D, self.D)))
        self.b = nn.Parameter(torch.randn(1, self.K, self.D))
        
        self.one_to_m = range(self.D)
        
    @property
    def σ(self):
        return F.softplus(self.pre_σ)
    
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
        samples, log_probs = core.sylvester.batch_sample_from_flow(
            self.μ, self.σ, self.Q, self.R1, self.R2, self.b, num_samples
        )    
        return samples[0], log_probs[0]
    
    def sample_from_each_layer(self, num_samples):
        samples_from_each_layer = core.sylvester.batch_sample_from_flow(
            self.μ, self.σ, self.Q, self.R1, self.R2, self.b, num_samples, return_z_from_each_layer=True
        )
        return samples_from_each_layer[0]

    
class ConditionalTriangularSylvesterFlow(nn.Module):

    pass

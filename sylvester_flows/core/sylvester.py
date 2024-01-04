import torch
import torch.distributions as tdist


def dtanh(x):
    return 1 - (torch.tanh(x)) ** 2


def transform(z, Q, R1, R2, b):
    """
    This function implements Equation 13 and 14 in the paper.
    
    Input shapes:
    
    z: (D)
    Q: (D, M)
    R1: (M, M)
    R2: (M, M)
    b: (M)
    
    Output shapes:
    
    (D+1)
    """
    
    pre_activation = R2 @ Q.T @ z + b
    
    new_z = z + Q @ R1 @ torch.tanh(pre_activation)
    log_abs_det = (1 + dtanh(pre_activation) * torch.diagonal(R2) * torch.diagonal(R1)).abs().log().sum().unsqueeze(0)
    
    return torch.cat([new_z, log_abs_det])


"""
batch_transform

Input shapes:
 
z: (bs, num_samples, D)
Q: (bs, D, M)
R1: (bs, M, M)
R2: (bs, M, M)
b: (bs, M)

Output shapes:

(bs, num_samples, D+1)

How it works:

after outer wrapper
(num_samples, D)
(D, M)
(M, M)
(M, M)
(M)

after inner wrapper

(D)
(D, M)
(M, M)
(M, M)
(M)

This code also works (but in a different way):

torch.vmap(
    torch.vmap(transform, (0, 0, 0, 0, 0), 0),
    (1, None, None, None, None),
    1
)
"""

batch_transform = torch.vmap(
    torch.vmap(transform, (0, None, None, None, None), 0), (0, 0, 0, 0, 0), 0
)


def batch_sample_from_flow(μ, σ, Q, R1, R2, b, num_samples, return_z_from_each_layer=False):
    """
    
    """
    
    bs, K, D, M = Q.shape

    # (you can't do vmap for sampling from distributions)
    q0 = tdist.Independent(tdist.Normal(loc=μ, scale=σ), reinterpreted_batch_ndims=1)  
    
    z = q0.rsample(sample_shape=torch.Size([num_samples]))  # (num_samples, bs, D)
    log_prob = q0.log_prob(z)  # (num_samples, bs)
    
    z = z.transpose(0, 1)  # (bs, num_samples, D)
    log_prob = log_prob.T  # (bs, num_samples)
    
    z_from_each_layer = torch.zeros(bs, K+1, num_samples, D)
    z_from_each_layer[:, 0, :, :] = z
  
    for k in range(K):
        
        out = batch_transform(
            z, Q[:, k, :, :], R1[:, k, :, :], R2[:, k, :, :], b[:, k, :]
        )
        
        z, logabsdet = out[:, :, :-1], out[:, :, -1]
        
        log_prob -= logabsdet
        z_from_each_layer[:, k+1, :, :] = z
    
    if return_z_from_each_layer:
        return z_from_each_layer
    else:
        return z, log_prob

import torch
import torch.nn as nn
import torch.distributions as tdist


def dtanh(x):
    return 1 - (torch.tanh(x)) ** 2


def transform(z, u, w, b):
    """
    This function implements Equation 10, 11 and 12 in the paper.

    Input shapes:

    z: (D)
    u: (D)
    w: (D)
    b: (1)

    Output shapes:

    (D+1)
    """

    pre_activation = torch.dot(w, z) + b

    new_z = z + u * torch.tanh(pre_activation)

    phi_z = dtanh(pre_activation) * w
    log_abs_det = (1 + torch.dot(u, phi_z)).abs().log().unsqueeze(0)

    return torch.cat([new_z, log_abs_det])


"""
batch_transform

Input shapes:
 
z: (bs, num_samples, D)
u: (bs, D)
w: (bs, D)
b: (bs, 1)

Output shapes:

(bs, num_samples, D+1)

How it works:

after outer wrapper
(num_samples, D)
(D)
(D)
(1)

after inner wrapper

(D)
(D)
(D)
(1)

This code also works (but in a different way):

torch.vmap(
    torch.vmap(sylvester_transform, (0, 0, 0, 0, 0), 0),
    (1, None, None, None, None),
    1
)
"""

batch_transform = torch.vmap(
    torch.vmap(transform, (0, None, None, None), 0), (0, 0, 0, 0), 0
)


def batch_sample_from_flow(μ, σ, u, w, b, num_samples, return_z_from_each_layer=False):
    """
    
    """

    bs, K, D = u.shape

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
            z, u[:, k, :], w[:, k, :], b[:, k, :]
        )
        
        z, logabsdet = out[:, :, :-1], out[:, :, -1]
        
        log_prob -= logabsdet
        z_from_each_layer[:, k+1, :, :] = z
    
    if return_z_from_each_layer:
        return z_from_each_layer
    else:
        return z, log_prob


def m(x):
    return -1 + nn.functional.softplus(x)


def compute_uhat(u, w):
    w_dot_u = torch.dot(u, w)
    uhat = u + (m(w_dot_u + 0.5413) - w_dot_u) * w / w.pow(2).sum()
    return uhat


batch_compute_uhat = torch.vmap(
    torch.vmap(compute_uhat, (0, 0), 0), (0, 0), 0
)

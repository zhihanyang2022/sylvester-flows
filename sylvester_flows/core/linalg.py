import torch


def batch_construct_orthogonal(q):
    """
    Copied the official codebase because it's well-written.
    
    Batch orthogonal matrix construction.
    :param q:  q contains batches of matrices, shape : (batch_size * num_flows, z_size * num_ortho_vecs)
    :return: batches of orthogonalized matrices, shape: (batch_size * num_flows, z_size, num_ortho_vecs)
    """

    # Reshape to shape (num_flows * batch_size, z_size * num_ortho_vecs)
    
    batch_size, num_flows, z_size, num_ortho_vecs = q.shape
    
    q = q.reshape(-1, z_size * num_ortho_vecs)

    norm = torch.norm(q, p=2, dim=1, keepdim=True)
    amat = torch.div(q, norm)
    dim0 = amat.size(0)
    amat = amat.reshape(dim0, z_size, num_ortho_vecs)

    max_norm = 0.
    
    identity = torch.eye(num_ortho_vecs).unsqueeze(0)

    # Iterative orthogonalization
    
    for s in range(30):
        
        tmp = torch.bmm(amat.transpose(2, 1), amat)
        tmp = identity - tmp
        tmp = identity + 0.5 * tmp
        amat = torch.bmm(amat, tmp)

        # Testing for convergence
        
        test = torch.bmm(amat.transpose(2, 1), amat) - identity
        norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
        norms = torch.sqrt(norms2)
        max_norm = torch.max(norms)
        
        if max_norm <= 1e-6:
            break

    if max_norm > 1e-6:
        print('\nWARNING WARNING WARNING: orthogonalization not complete')
        print('\t Final max norm =', max_norm)

        print()

    # Reshaping: first dimension is batch_size
    amat = amat.reshape(batch_size, num_flows, z_size, num_ortho_vecs)

    return amat


def construct_householder(vs):
    num_vs, v_dim = vs.shape

    I = torch.eye(v_dim, v_dim)

    P = I

    for i in range(num_vs):
        v = vs[i].view(-1, 1)
        v = v / torch.linalg.vector_norm(v)

        P = (I - 2 * v @ v.T) @ P

    return P


batch_construct_householder = torch.vmap(torch.vmap(construct_householder, (0, ), 0), (0, ), 0)

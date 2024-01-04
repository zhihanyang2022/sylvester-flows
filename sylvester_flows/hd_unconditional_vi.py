"""
hd_unconditional_vi.py

Run 5 seeds for a flow type with specific parameters.
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle
import argparse
import os

from flow import UnconditionalPlanarFlow, UnconditionalOrthogonalSylvesterFlow, UnconditionalHouseholderSylvesterFlow, \
    UnconditionalTriangularSylvesterFlow, UnconditionalRandomSylvesterFlow, UnconditionalIdentitySylvesterFlow


# ==================================================
# potential func
# ==================================================

torch.manual_seed(42)

key_point_1 = torch.rand(1, 10) - 0.5
key_point_2 = torch.rand(1, 10) - 0.5
key_point_3 = torch.rand(1, 10) - 0.5
key_point_4 = torch.rand(1, 10) - 0.5
key_point_5 = torch.rand(1, 10) - 0.5
key_point_6 = torch.rand(1, 10) - 0.5
key_point_7 = torch.rand(1, 10) - 0.5
key_point_8 = torch.rand(1, 10) - 0.5
key_point_9 = torch.rand(1, 10) - 0.5
key_point_10 = torch.rand(1, 10) - 0.5


# def log_unnormalized_density(x):
#     div = 0.2
#     return torch.logsumexp(torch.hstack([
#         - torch.linalg.vector_norm(x - key_point_1, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_2, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_3, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_4, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_5, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_6, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_7, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_8, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_9, dim=1).div(div).pow(2).view(-1, 1),
#         - torch.linalg.vector_norm(x - key_point_10, dim=1).div(div).pow(2).view(-1, 1)
#     ]), dim=1)


def log_unnormalized_density(x):
    variance = 0.04
    return torch.logsumexp(torch.hstack([
        - torch.linalg.vector_norm(x - key_point_1, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_2, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_3, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_4, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_5, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_6, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_7, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_8, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_9, dim=1).pow(2).div(variance).view(-1, 1),
        - torch.linalg.vector_norm(x - key_point_10, dim=1).pow(2).div(variance).view(-1, 1)

    ]), dim=1)


# ==================================================
# parse arguments
# ==================================================

parser = argparse.ArgumentParser()
parser.add_argument("flow_type", help="The type of flow to use")
parser.add_argument("num_layer", help="Number of layers to use within the flow", type=int)
parser.add_argument("-M", help="[Only for orthogonal Sylvester] number of orthogonal columns", type=int)
parser.add_argument("-H", help="[Only for Householder Sylvester] number of Householder vectors", type=int)
args = parser.parse_args()
config = vars(args)

flow_type = config["flow_type"]
num_layer = config["num_layer"]
M = config["M"]
H = config["H"]

if flow_type == "planar":

    assert (M is None) and (H is None)

    def get_q():
        return UnconditionalPlanarFlow(D=10, K=num_layer)
    folder = f"./saved/hd_unconditional_vi/planar_K{num_layer}"

elif flow_type == "orthogonal":

    assert (M is not None) and (H is None)

    def get_q():
        return UnconditionalOrthogonalSylvesterFlow(D=10, M=M, K=num_layer)
    folder = f"./saved/hd_unconditional_vi/orthogonal_K{num_layer}_M{M}"

elif flow_type == "householder":

    assert (M is None) and (H is not None)

    def get_q():
        return UnconditionalHouseholderSylvesterFlow(D=10, H=H, K=num_layer)
    folder = f"./saved/hd_unconditional_vi/householder_K{num_layer}_H{H}"

elif flow_type == "triangular":

    assert (M is None) and (H is None)

    def get_q():
        return UnconditionalTriangularSylvesterFlow(D=10, K=num_layer)
    folder = f"./saved/hd_unconditional_vi/triangular_K{num_layer}"

elif flow_type == "random":

    assert (M is None) and (H is None)

    def get_q():
        return UnconditionalRandomSylvesterFlow(D=10, K=num_layer)
    folder = f"./saved/hd_unconditional_vi/random_K{num_layer}"

elif flow_type == "identity":

    assert (M is None) and (H is None)

    def get_q():
        return UnconditionalIdentitySylvesterFlow(D=10, K=num_layer)

    folder = f"./saved/hd_unconditional_vi/identity_K{num_layer}"

else:

    raise ValueError()


# ==================================================
# do variational inference
# ==================================================

def main(seed):

    print(f"========== Seed {seed} ==========")

    folder_seed = os.path.join(folder, f"seed{seed}")
    os.makedirs(folder_seed, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    q = get_q()
    opt = optim.Adam(q.parameters(), lr=2e-3)

    # stochastic variational inference

    kls = []

    for i in range(1, 20001):

        samples, logps = q.sample_and_compute_logp(1000)
        kl = (logps - log_unnormalized_density(samples)).mean()

        kls.append(float(kl))

        opt.zero_grad()
        kl.backward()
        opt.step()

        if i % 100 == 0:
            print(i, float(kl))

    # save trained model

    torch.save(q.state_dict(), os.path.join(folder_seed, "q.pth"))

    # save training history

    with open(os.path.join(folder_seed, "kls"), "wb") as fp:
        pickle.dump(kls, fp)

    # generate samples

    with torch.no_grad():
        samples, _ = q.sample_and_compute_logp(100000)
        samples = samples.numpy()

    # plot samples and save the plot

    fig = plt.figure(figsize=(10, 10), )

    for i in range(1, 10 + 1):
        for j in range(1, 10 + 1):
            if i < j:
                fig.add_subplot(10, 10, (i - 1) * 10 + j)

                plt.hist2d(samples[:, i - 1], samples[:, j - 1], bins=50, range=[[-1, 1], [-1, 1]])

                plt.xlim(-1, 1)
                plt.ylim(-1, 1)
                plt.xticks([])
                plt.yticks([])
                plt.gca().set_aspect("equal")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(os.path.join(folder_seed, "learned_density.png"), dpi=100, bbox_inches='tight')


seeds = [1, 2, 3, 4, 5]

for s in seeds:
    main(s)

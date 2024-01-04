import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pickle

from flow import UnconditionalOrthogonalSylvesterFlow
from potential_functions import pick_potential_func

# ==================================================

parser = argparse.ArgumentParser()
parser.add_argument("u_index", help="Index of some potential function")
parser.add_argument("flow_type", help="The type of flow to use")
parser.add_argument("num_layer", help="Number of layers to use within the flow")
parser.add_argument("seed")
args = parser.parse_args()
config = vars(args)

flow_type = config["flow_type"]
num_layer = int(config["num_layer"])
u_index = int(config["u_index"])
seed = int(config["seed"])

torch.manual_seed(seed)

folder = f"../pngs/2d_unconditional_vi/U{u_index}_{flow_type}_{num_layer}_{seed}"
os.makedirs(folder, exist_ok=True)

U = pick_potential_func(u_index)

if flow_type == "orthogonal":
    flow = UnconditionalOrthogonalSylvesterFlow(D=2, M=2, K=num_layer)
else:
    raise ValueError(f"{flow_type} is not a valid flow.")

# ==================================================

opt = optim.Adam(flow.parameters(), lr=1e-3)

kls = []

num_epochs = 5000

for i in range(1, num_epochs + 1):

    samples, logps = flow.sample_and_compute_logp(num_samples=1000)
    kl = (logps + U(samples)).mean()
    kls.append(float(kl))

    opt.zero_grad()
    kl.backward()
    opt.step()

    if i % 100 == 0:
        print(i, float(kl))

# ==================================================
# Training history
# ==================================================

with open(f"{folder}/kls", "wb") as fp:
    pickle.dump(kls, fp)

# with open(f"{folder}/kls", "rb") as fp:
#     kls = b = pickle.load(fp)
#
# print(kls)

# ==================================================
# Density plot
# ==================================================

fig = plt.figure(figsize=(3.75, 3.75))

with torch.no_grad():
    samples, _ = flow.sample_and_compute_logp(num_samples=int(1e6))

samples = samples.numpy()

plt.hist2d(samples[:, 0], samples[:, 1], bins=100, cmap="jet", range=[[-6, 6], [-6, 6]])

plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig(f"{folder}/learned_density.png", dpi=300, bbox_inches='tight')

# ==================================================
# Samples after each layer
# ==================================================

n = 10000

# copied from stackoverflow
phi = np.linspace(0, 2*np.pi, n)
x = np.sin(phi)
y = np.cos(phi)
rgb_cycle = (np.stack((np.cos(phi          ), # Three sinusoids,
                       np.cos(phi+2*np.pi/3), # 120° phase shifted,
                       np.cos(phi-2*np.pi/3)
                      )).T # Shape = (60,3)
             + 1)*0.5

with torch.no_grad():
    samples_from_each_layer = flow.sample_from_each_layer(n)
samples_from_each_layer = [samples.numpy() for samples in samples_from_each_layer]

indices = np.argsort(
    (samples_from_each_layer[0][:,0] - float(flow.μ[0, 0])) ** 2 +
    (samples_from_each_layer[0][:,1] - float(flow.μ[0, 1])) ** 2
)

fig = plt.figure(figsize=(6, 1))
for i, j in enumerate(np.round(np.linspace(0, num_layer, 6))):
    j = int(j)
    fig.add_subplot(1, 6, i+1)
    plt.scatter(samples_from_each_layer[j][:,0][indices],
                samples_from_each_layer[j][:,1][indices],
                color=rgb_cycle,
                s=0.01, alpha=1)
    plt.text(-4.5, 3, f"{j}", size=13)
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xticks([])
    plt.yticks([])
fig.subplots_adjust(wspace=0, hspace=0)

plt.savefig(f"{folder}/samples_from_each_layer.png", dpi=300, bbox_inches='tight')

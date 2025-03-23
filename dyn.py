# Pranav Minasandra

import matplotlib as mpl
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import scipy.spatial
import torch

mpl.use("TkAgg")
import config

dt = config.DT
g = config.CONST_GRAV
c = config.CONST_DAMP

# NOTE: vectors are in n×d matrix.
def pairwise_distances_and_directions(X, eps=1e-9):
    diff = X[:, None, :] - X[None, :, :]
    dist = torch.norm(diff, dim=-1)
    unit = diff / (dist[..., None] + eps)
    return dist, -unit# distance matrix (n×n), unit vectors (n×n×d)

def step(X, V, M, K, L):
    distances, unit_vectors = pairwise_distances_and_directions(X)

    force_magnitudes = K*(distances - L)
    spring_force_vectors = force_magnitudes[..., None] * unit_vectors
    damping_force_vectors = -c*V

    net_forces = spring_force_vectors.sum(dim=1) + damping_force_vectors
    net_forces += g*M[:, None]
#    print(net_forces/ M[:, None])
    V_new = V + (net_forces/M[:, None])*dt
    X_new = X + V_new*dt

    below_ground = X_new[:, -1] < 0.0 #can't go below the ground
    X_new[below_ground, -1] = 0.0
    V[below_ground, -1] *= -0.99
    return X_new, V_new

def animate(X, V, M, K, L, fps=50, tot_time=10):

    iters_per_sec = 1/dt
    update_every = int(iters_per_sec/fps)

    fig, ax = plt.subplots()
    sc = ax.scatter(X[:, 0], X[:, 1], s=5.0,  color='black')
    ax.set_xlim([-1, 11])
    ax.set_ylim([-1, 11])

    def update(i):
        nonlocal X, V, M, K, L
        for j in range(update_every):
            X, V = step(X, V, M, K, L)
            sc.set_offsets(X)
            ax.set_xlim([-1, 11])
            ax.set_ylim([-1, 11])
        return ax,

    ani = FuncAnimation(fig, update, frames = int(fps*tot_time), interval=(1/fps)*1000, blit=False)
    return ani

if __name__ == "__main__":
#    X = torch.tensor([[0.0, 0], [1, 0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2]]) + torch.tensor([0, 0.5])
##    V = torch.tensor([[0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0], [0,0]])
#    V = torch.rand(size=(9, 2))*0.0
##    V[5, 0] += 0.5
#    K, _ = pairwise_distances_and_directions(X)
#    K[K > 1.1] = 0
#    K *= 100.0
#    L, _ = pairwise_distances_and_directions(X)
#    L[L > 1.1] = 0
#    M = torch.tensor([1.0, 1, 1, 1, 1, 1, 1, 1, 1])

#    X = torch.tensor([[0.0, 1], [1, 1]])
#    V = torch.tensor([[0.0, 0], [0, 0]])
#
#    K = torch.tensor([[0.0, 1], [1, 0]])*100
#    L = torch.tensor([[0.0, 2], [2, 0]])
#    M = torch.tensor([1.0, 1])

    STEP_SIZE = 0.25
    x_vals = torch.arange(0, 5, STEP_SIZE)
    y_vals = torch.arange(0, 5, STEP_SIZE) 

# Create 2D grid
    X_grid, Y_grid = torch.meshgrid(x_vals, y_vals, indexing='ij')  # both (5, 4)
    X = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=1)  # (20, 2)

    X = X[(X[:, 1] - torch.abs(X[:, 0] - 2)) > 1.8]
    V = torch.zeros(size=(X.shape))
    V += torch.tensor([0.4, 0.4]) + torch.rand(size=(V.shape))*0.1

    V[X[:, 0] < 2] += torch.tensor([0, 1.0])
    V[X[:, 0] > 2] += torch.tensor([0, -1.0])
    K, _ = pairwise_distances_and_directions(X)
    Kc = K.clone().detach()
    Kc[K < 1.5*STEP_SIZE] = 1
    Kc[K > 1.5*STEP_SIZE] = 0
    K = Kc*250.0
    K.diagonal().fill_(0.0)

    K[X[:, 1] < 2] *= 0.3
    print(K)
    L, _ = pairwise_distances_and_directions(X)
    L[L > 1.5*STEP_SIZE] = 0
    L.diagonal().fill_(0.0)
    M = torch.ones(X.shape[0])*0.1

# Stack into shape (n, 2) = (5*4, 2)
    ani = animate(X, V, M, K, L, fps=30, tot_time=5)
#    plt.show()
    ani.save("animation.gif", writer='ffmpeg')


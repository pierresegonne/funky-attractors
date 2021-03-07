import matplotlib.pyplot as plt
import numpy as np

from attractors import *

# Manual parameters
n_steps = 1000000# 5 * int(1e5)
params = {"a": -1.4, "b": 1.6, "c": 1.0, "d": 0.7}
params = None
X_init = np.array([0.1, 0., 0.1])

# clifford, de_jong, ikeda, lorentz, rossler
attractor = ROSSLER
attractor_f, params = get_attractor_function(attractor, params)


def compute_states(X, attractor, n_steps, params):

    for i in range(n_steps):
        X[i+1] = attractor(X[i]) if params is None else attractor(X[i], params)

    return X

def plot_2D_attractor(X, params):

    plt.figure(figsize=(5,5))
    plt.plot(X[:, 0], X[:, 1], '.', color='black', alpha=0.2, markersize=0.2)
    plt.axis("off")

    plt.savefig(f"plots/{attractor}_{str(params)}.png", dpi = 800, pad_inches = 0, bbox_inches = 'tight', facecolor='white')
    # SVG too big using point by point approach
    # plt.savefig(f"plots/{attractor}_{str(params)}.svg", pad_inches = 0, bbox_inches = 'tight', facecolor='white')
    plt.close()

def plot_3D_attractor(X, params):
    fig = plt.figure(figsize=(5,5))
    ax = fig.gca(projection='3d')

    plt.plot(X[:, 0], X[:, 1], X[:, 2], '.', color='black', alpha=0.2, markersize=0.2)

    plt.savefig(f"plots/{attractor}_{str(params)}.png", dpi = 800, pad_inches = 0, bbox_inches = 'tight', facecolor='white')
    plt.close()

if __name__ == "__main__":

    D = X_init.shape[0]
    X = np.empty((n_steps + 1, D))
    X[0] = X_init
    X = compute_states(X, attractor_f, n_steps, params)

    if D == 2:
        plot_2D_attractor(X, params)
    elif D == 3:
        plot_3D_attractor(X, params)

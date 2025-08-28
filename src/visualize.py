# Utilities to project high-dimensional features (global intention, sub-intentions, actions) to 2D
# and visualize them as requested.
#
# How to use (example):
#   from pathlib import Path
#   import numpy as np
#   # Suppose you already have:
#   #   G: (B, D_clip), S: (B, T, D_clip), A: (B, T, D_clip)
#   # plot_intentions_2d(G, S, A, batch_idx=0, method='pca', cmap_name='viridis')
#
# Notes:
# - Uses PCA via SVD (no external dependencies).
# - Colors encode time index t in [0, T-1]. Same t across sub-intention and action share exactly the same color.
# - Global intention: one large circle.
# - Sub-intentions: filled circles.
# - Actions: hollow circles (same color as sub-intentions at the same t).
# - One figure per batch index; call multiple times if you want multiple batches.
#
# If you want to save the plot, pass save_path=".../figure.png".

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

def save_matrix_npy(matrix: np.ndarray, index=0, filename='./src/output_action/'):
    """
    Save a matrix of shape (S, 1, T, D) into .npy format.

    Args:
        matrix (np.ndarray): Input array with shape (S, 1, T, D).
        filename (str): Path where the .npy file will be saved (e.g., "output.npy").
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy ndarray.")
    if matrix.ndim != 4:
        raise ValueError(f"Expected a 4D array of shape (S, 1, T, D), got shape {matrix.shape}")

    np.save(filename+str(index)+'.png', matrix)
    print(f"Saved matrix with shape {matrix.shape} to {filename}")



def _pca_2d(X: np.ndarray) -> np.ndarray:
    """
    Center X and return the first two principal component projections.
    X: (N, D)
    Returns: Z (N, 2)
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S V^T; first two PCs are columns of V[:, :2]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    V2 = Vt[:2].T  # (D, 2)
    Z = Xc @ V2    # (N, 2)
    return Z


def reduce_to_2d(X: np.ndarray, method: str = "pca") -> np.ndarray:
    """
    Reduce high-dim features to 2D.
    Currently supports only PCA (deterministic).
    """
    method = method.lower()
    if method == "pca":
        return _pca_2d(X)
    else:
        raise ValueError(f"Unknown method '{method}'. Supported: 'pca'.")


def plot_intentions_2d_GSA(
    global_intentions: np.ndarray,  # (B, D)
    sub_intentions: np.ndarray,     # (B, T, D)
    actions: np.ndarray,            # (B, T, D)
    png_name,
    batch_idx: int = 0,
    method: str = "pca",
    cmap_name: str = "viridis",
    figsize: Tuple[int, int] = (7, 7),
    title: Optional[str] = None,
    save_path='./src/visualizations-GSA/',
    show: bool = False,
) -> np.ndarray:
    """
    Projects one batch's features into 2D and makes a single plot.
    Returns the 2D coordinates as a dict-like np array with fields:
      coords_global (1,2), coords_sub (T,2), coords_actions (T,2)
    """
    # Basic checks
    if global_intentions.ndim != 2:
        raise ValueError("global_intentions must be (B, D)")
    if sub_intentions.ndim != 3:
        raise ValueError("sub_intentions must be (B, T, D)")
    if actions.ndim != 3:
        raise ValueError("actions must be (B, T, D)")
    if not (global_intentions.shape[0] == sub_intentions.shape[0] == actions.shape[0]):
        raise ValueError("Batch size (B) must match across inputs")
    if not (sub_intentions.shape[1] == actions.shape[1]):
        raise ValueError("Time length T must match between sub_intentions and actions")
    if not (global_intentions.shape[1] == sub_intentions.shape[2] == actions.shape[2]):
        raise ValueError("Feature dim D must match across inputs")

    B, T, D = sub_intentions.shape
    if not (0 <= batch_idx < B):
        raise IndexError(f"batch_idx {batch_idx} out of range [0, {B-1}]")

    g = global_intentions[batch_idx:batch_idx+1]  # (1, D)
    s = sub_intentions[batch_idx]                 # (T, D)
    a = actions[batch_idx]                        # (T, D)

    # Stack for joint projection
    X = np.concatenate([g, s, a], axis=0)  # (1 + T + T, D) = (1 + 2T, D)
    Z = reduce_to_2d(X, method=method)     # (1 + 2T, 2)

    # Split back
    z_g   = Z[0:1]          # (1, 2)
    z_s   = Z[1:1+T]        # (T, 2)
    z_act = Z[1+T:1+2*T]    # (T, 2)

    # Colormap for time indices (same color used for sub & action at same t)
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=max(T-1, 1))  # avoid div by zero

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot global intention: big circle
    ax.scatter(z_g[:, 0], z_g[:, 1], marker='o', s=180, edgecolors='black', linewidths=1.5, facecolors='none', label='Global')

    # Plot sub-intentions (filled circles) and actions (hollow circles) per time with shared color
    for t in range(T):
        color = cmap(norm(t))
        # sub-intention: filled circle
        ax.scatter(z_s[t, 0], z_s[t, 1], marker='o', s=60, c=[color], edgecolors='black', linewidths=0.5)
        # action: hollow circle (same color)
        ax.scatter(z_act[t, 0], z_act[t, 1], marker='o', s=60, facecolors='none', edgecolors=[color], linewidths=1.5)

    # Cosmetics
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if title is None:
        ax.set_title(f"Batch {batch_idx} — Global (big), Sub (filled), Action (hollow); colors = time index")
    else:
        ax.set_title(title)

    # Legend (proxy artists)
    legend_elems = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='black', markersize=10, linewidth=0, label='Global (circle)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='gray', markeredgecolor='black', markersize=7, linewidth=0, label='Sub-intention (filled)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='gray', markersize=7, linewidth=0, label='Action (hollow)'),
    ]
    ax.legend(handles=legend_elems, loc='best', frameon=True)

    # Colorbar for time index
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time index t")

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')

    if save_path is not None:
        fig.savefig(save_path + png_name, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Return coordinates for potential further processing
    out = {
        "coords_global": z_g,
        "coords_sub": z_s,
        "coords_actions": z_act
    }
    return out

def plot_intentions_2d_GS(
    global_intentions: np.ndarray,  # (B, D)
    sub_intentions: np.ndarray,     # (B, T, D)
    actions: np.ndarray,            # (B, T, D)
    png_name,
    batch_idx: int = 0,
    method: str = "pca",
    cmap_name: str = "viridis",
    figsize: Tuple[int, int] = (7, 7),
    title: Optional[str] = None,
    save_path='./src/visualizations-GS/',
    show: bool = False,
) -> np.ndarray:
    """
    Projects one batch's features into 2D and makes a single plot.
    Returns the 2D coordinates as a dict-like np array with fields:
      coords_global (1,2), coords_sub (T,2), coords_actions (T,2)
    """
    # Basic checks
    if global_intentions.ndim != 2:
        raise ValueError("global_intentions must be (B, D)")
    if sub_intentions.ndim != 3:
        raise ValueError("sub_intentions must be (B, T, D)")
    if actions.ndim != 3:
        raise ValueError("actions must be (B, T, D)")
    if not (global_intentions.shape[0] == sub_intentions.shape[0] == actions.shape[0]):
        raise ValueError("Batch size (B) must match across inputs")
    if not (sub_intentions.shape[1] == actions.shape[1]):
        raise ValueError("Time length T must match between sub_intentions and actions")
    if not (global_intentions.shape[1] == sub_intentions.shape[2] == actions.shape[2]):
        raise ValueError("Feature dim D must match across inputs")

    B, T, D = sub_intentions.shape
    if not (0 <= batch_idx < B):
        raise IndexError(f"batch_idx {batch_idx} out of range [0, {B-1}]")

    g = global_intentions[batch_idx:batch_idx+1]  # (1, D)
    s = sub_intentions[batch_idx]                 # (T, D)
    #a = actions[batch_idx]                        # (T, D)

    # Stack for joint projection
    X = np.concatenate([g, s], axis=0)  # (1 + T + T, D) = (1 + 2T, D)
    Z = reduce_to_2d(X, method=method)     # (1 + 2T, 2)

    # Split back
    z_g   = Z[0:1]          # (1, 2)
    z_s   = Z[1:1+T]        # (T, 2)
    #z_act = Z[1+T:1+2*T]    # (T, 2)

    # Colormap for time indices (same color used for sub & action at same t)
    cmap = get_cmap(cmap_name)
    norm = Normalize(vmin=0, vmax=max(T-1, 1))  # avoid div by zero

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot global intention: big circle
    ax.scatter(z_g[:, 0], z_g[:, 1], marker='o', s=180, edgecolors='black', linewidths=1.5, facecolors='none', label='Global')

    # Plot sub-intentions (filled circles) and actions (hollow circles) per time with shared color
    for t in range(T):
        color = cmap(norm(t))
        # sub-intention: filled circle
        ax.scatter(z_s[t, 0], z_s[t, 1], marker='o', s=60, c=[color], edgecolors='black', linewidths=0.5)
        # action: hollow circle (same color)
        #ax.scatter(z_act[t, 0], z_act[t, 1], marker='o', s=60, facecolors='none', edgecolors=[color], linewidths=1.5)

    # Cosmetics
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    if title is None:
        ax.set_title(f"Batch {batch_idx} — Global (big), Sub (filled), Action (hollow); colors = time index")
    else:
        ax.set_title(title)

    # Legend (proxy artists)
    legend_elems = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='black', markersize=10, linewidth=0, label='Global (circle)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='gray', markeredgecolor='black', markersize=7, linewidth=0, label='Sub-intention (filled)'),
        #Line2D([0], [0], marker='o', color='none', markerfacecolor='none', markeredgecolor='gray', markersize=7, linewidth=0, label='Action (hollow)'),
    ]
    ax.legend(handles=legend_elems, loc='best', frameon=True)

    # Colorbar for time index
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time index t")

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    ax.set_aspect('equal', adjustable='datalim')

    if save_path is not None:
        fig.savefig(save_path + png_name, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

    # Return coordinates for potential further processing
    out = {
        "coords_global": z_g,
        "coords_sub": z_s,
    }
    return out


# --- Optional: Tiny demo with random data (commented out) ---
# B, T, D = 2, 12, 64
# G = np.random.randn(B, D)
# S = np.random.randn(B, T, D)
# A = np.random.randn(B, T, D)
# plot_intentions_2d(G, S, A, batch_idx=0, method='pca', cmap_name='viridis', title="Demo (Random)")

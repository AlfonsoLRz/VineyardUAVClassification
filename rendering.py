import matplotlib.pyplot as plt
import numpy as np
from randomness import *
import umap.umap_ as umap


def render_hc_spectrum_label(hc_numpy, mask):
    """
    Renders the spectrum of every label in the hypercube.
    """
    n_classes = np.unique(mask)

    for class_id in n_classes:
        sample_subset = hc_numpy[mask == class_id, :]
        plt.plot(np.average(sample_subset, axis=0), label=class_id)
    plt.legend()
    plt.show()


def render_mask_histogram(label):
    plt.hist(label.flatten(), bins='auto', rwidth=1.0)
    plt.show()


def render_umap_spectrum(patch, label):
    """
    Renders the UMAP of the spectrum of every label in the hypercube.
    """
    reducer = umap.UMAP(random_state=random_seed)
    embedding = reducer.fit_transform(patch)

    # Adjust size of plot
    plt.figure(figsize=(10, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.tight_layout()
    plt.show()

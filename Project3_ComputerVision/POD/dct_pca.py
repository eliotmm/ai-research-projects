#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCT-based reconstruction des blobs partiellement masqués
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread
from scipy.fftpack import dct, idct


def load_blob_dataset(folder: str) -> np.ndarray:
    paths = sorted(glob(f"{folder}/*.png"))
    images = [imread(p).astype(np.float32) for p in paths]
    return np.stack(images)


def dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')


def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')


def compute_dct_basis(H, W, n_components):
    """
    Génère une base DCT 2D directe, sans apprentissage SVD.
    """
    basis = []
    for u in range(H):
        for v in range(W):
            vec_u = dct(np.eye(H)[u], norm='ortho')
            vec_v = dct(np.eye(W)[v], norm='ortho')
            basis_uv = np.outer(vec_u, vec_v)
            basis.append(basis_uv)
            if len(basis) >= n_components:
                break
        if len(basis) >= n_components:
            break
    return np.array(basis)


def project_and_reconstruct(img, mask, basis):
    H, W = img.shape
    img_flat = img.flatten()
    mask_flat = mask.flatten().astype(bool)
    basis_flat = basis.reshape(basis.shape[0], -1)
    basis_obs = basis_flat[:, mask_flat]
    obs = img_flat[mask_flat]

    coeffs, *_ = np.linalg.lstsq(basis_obs.T, obs, rcond=None)
    recon_flat = np.dot(coeffs, basis_flat)
    recon_img = recon_flat.reshape(H, W)
    final_img = img.copy()
    final_img[~mask] = recon_img[~mask]
    return final_img, coeffs


def show_modes(modes, ncols=5):
    n_modes = len(modes)
    nrows = (n_modes + ncols - 1) // ncols
    plt.figure(figsize=(3 * ncols, 3 * nrows))
    for i, mode in enumerate(modes):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(mode, cmap="gray")
        plt.title(f"Mode {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    folder = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/single_blob_center"
    X = load_blob_dataset(folder)

    folder_masked = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/single_blob_center_masked_grid"
    X_masked = load_blob_dataset(folder_masked)

    H, W = X.shape[1], X.shape[2]
    n_components = 50
    basis = compute_dct_basis(H, W, n_components)

    show_modes(basis)

    for i in range(5):
        masked_img = X_masked[i]
        mask = masked_img != 0
        recon_img, _ = project_and_reconstruct(masked_img, mask, basis)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(masked_img, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Image masquée #{i}")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(recon_img, cmap="gray", vmin=0, vmax=255)
        plt.title("Reconstruction DCT")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
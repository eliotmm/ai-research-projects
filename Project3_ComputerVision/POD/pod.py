#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:43:20 2025

@author: eliotmorard
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from skimage.io import imread
import os
import uuid

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error

def compute_metrics(original, reconstructed, mask, display=False):
    # Mesures sur toute l’image
    mse_full = mean_squared_error(original, reconstructed)
    psnr_full = peak_signal_noise_ratio(original, reconstructed, data_range=255)
    ssim_full = structural_similarity(original, reconstructed, data_range=255)
    # Mesures sur la zone masquée uniquement
    mask_inv = ~mask
    mse_masked = mean_squared_error(original[mask_inv], reconstructed[mask_inv])
    if mse_masked == 0:
        psnr_masked = float('inf')
    else:
        psnr_masked = 10 * np.log10(255**2 / mse_masked)
    # SSIM masqué (nécessite que la zone masquée forme un rectangle assez grand)
    # On peut parfois simplement mettre np.nan si ce n’est pas applicable ou prendre un patch carré
    try:
        # Option 1 : Si tu veux le SSIM juste sur la zone masquée (si elle est assez grande et rectangulaire)
        ssim_masked = structural_similarity(
            original, reconstructed,
            data_range=255,
            win_size=7,
            gaussian_weights=True,
            use_sample_covariance=False,
            multichannel=False,
            mask=mask_inv
        )
    except Exception as e:
        ssim_masked = np.nan

    if display:
        print(f"MSE (full): {mse_full:.5f}, MSE (masked): {mse_masked:.5f}")
        print(f"PSNR (full): {psnr_full:.2f} dB, PSNR (masked): {psnr_masked:.2f} dB")
        print(f"SSIM (full): {ssim_full:.3f}, SSIM (masked): {ssim_masked:.3f}")

    return dict(
        mse_full=mse_full,
        mse_masked=mse_masked,
        psnr_full=psnr_full,
        psnr_masked=psnr_masked,
        ssim_full=ssim_full,
        ssim_masked=ssim_masked
    )

def load_blob_dataset(folder: str) -> np.ndarray:
    paths = sorted(glob(f"{folder}/*.png"))
    images = [imread(p).astype(np.float32) for p in paths]
    return np.stack(images)

def compute_pod(X: np.ndarray, n_components: int = 15):
    """
    Applique la méthode POD (SVD) sur un ensemble d'images de taille (N, H, W)
    Retourne :
        - modes (n_components, H, W)
        - valeurs singulières
        - moyenne
    """
    N, H, W = X.shape
    X_flat = X.reshape(N, H*W)
    mean = np.mean(X_flat, axis=0)
    X_centered = X_flat - mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    modes = Vt[:n_components].reshape(n_components, H, W)

    return modes, S[:n_components], mean.reshape(H, W)

# def reconstruct_image(img: np.ndarray, modes: np.ndarray, mean: np.ndarray):
#     """
#     Reconstruit une image partielle (H, W) projetée sur les modes POD.
#     """
#     H, W = img.shape
#     img_flat = img.flatten()
#     mean_flat = mean.flatten()
#     centered = img_flat - mean_flat

#     # projection coefficients
#     coeffs = np.dot(modes.reshape(modes.shape[0], -1), centered)
#     recon_flat = mean_flat + np.dot(coeffs, modes.reshape(modes.shape[0], -1))
#     return recon_flat.reshape(H, W), coeffs

def show_modes(modes, ncols=5):
    """
    Affiche les modes avec `ncols` subplots par ligne.
    """
    n_modes = len(modes)
    nrows = (n_modes + ncols - 1) // ncols  # arrondi supérieur
    plt.figure(figsize=(3 * ncols, 3 * nrows))

    for i, mode in enumerate(modes):
        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(mode, cmap="gray")
        plt.title(f"Mode {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
    

def plot_pod_energy_log_scale(S, max_components=None):
    """
    Affiche un barplot propre en log-échelle de l'énergie expliquée par mode (entiers propres).
    """
    energy = S**2
    energy_ratio = energy / np.sum(energy)

    if max_components is not None:
        energy_ratio = energy_ratio[:max_components]

    modes = np.arange(0, len(energy_ratio))

    plt.figure(figsize=(12, 8))
    plt.bar(modes, energy_ratio, width=0.8)

    # Axe log en y
    plt.yscale('log')
    plt.xlabel("Mode")
    plt.ylabel("Énergie expliquée (log)")
    plt.title("Énergie par mode (log-échelle)")

    # Ticks entiers sur l'axe x
    plt.xticks(modes)  # Affiche chaque mode (1, 2, 3, ...)
    plt.grid(True, which="both", axis='y', linestyle='-', color='black', linewidth=0.1, alpha=1)

    plt.tight_layout()
    plt.show()
    

# def reconstruct_missing_only(img: np.ndarray, mask: np.ndarray, modes: np.ndarray, mean: np.ndarray):
#     """
#     Reconstruit uniquement les pixels masqués d'une image partielle à l'aide des modes POD.
#     """
#     H, W = img.shape
#     img_flat = img.flatten()
#     mean_flat = mean.flatten()
#     mask_flat = mask.flatten().astype(bool)  # 1 pour non masqué, 0 pour masqué

#     # Partie observée : on soustrait la moyenne
#     obs = img_flat[mask_flat] - mean_flat[mask_flat]

#     # Projection POD
#     modes_flat = modes.reshape(modes.shape[0], -1)
#     modes_obs = modes_flat[:, mask_flat]

#     # Résolution des coefficients par moindres carrés
#     coeffs, *_ = np.linalg.lstsq(modes_obs.T, obs, rcond=None)

#     # Reconstruction totale
#     recon_flat = mean_flat + np.dot(coeffs, modes_flat)

#     # Fusion avec l'image connue : on garde les pixels observés
#     final_flat = img_flat.copy()
#     final_flat[~mask_flat] = recon_flat[~mask_flat]  # remplacer seulement les masqués

#     return final_flat.reshape(H, W), coeffs

def reconstruct_missing_only(img: np.ndarray, mask: np.ndarray, modes: np.ndarray, mean: np.ndarray, lmbda=0.0):
    """
    Reconstruit uniquement les pixels masqués d'une image partielle à l'aide des modes POD,
    avec option de régularisation (ridge/Tikhonov).
    """
    H, W = img.shape
    img_flat = img.flatten()
    mean_flat = mean.flatten()
    mask_flat = mask.flatten().astype(bool)

    obs = img_flat[mask_flat] - mean_flat[mask_flat]
    modes_flat = modes.reshape(modes.shape[0], -1)
    modes_obs = modes_flat[:, mask_flat]

    # Moindres carrés régularisés
    A = modes_obs.T
    b = obs
    if lmbda > 0:
        ATA = A.T @ A + lmbda * np.eye(A.shape[1])
        ATb = A.T @ b
        coeffs = np.linalg.solve(ATA, ATb)
    else:
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)

    recon_flat = mean_flat + np.dot(coeffs, modes_flat)
    final_flat = img_flat.copy()
    final_flat[~mask_flat] = recon_flat[~mask_flat]
    return final_flat.reshape(H, W), coeffs


def load_test_pairs(folder_complete, folder_masked):
    """
    Charge les paires (original, masquée) ayant le même nom.
    Retourne une liste de tuples (img_complete, img_masked)
    """
    files = sorted(os.listdir(folder_complete))
    pairs = []
    for fname in files:
        complete_path = os.path.join(folder_complete, fname)
        masked_path = os.path.join(folder_masked, fname)
        if os.path.exists(masked_path):
            img_complete = imread(complete_path).astype(np.float32)
            img_masked = imread(masked_path).astype(np.float32)
            pairs.append((img_complete, img_masked))
    return pairs
    
# if __name__ == "__main__":
#     folder = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/single_blob_center"
#     X = load_blob_dataset(folder)
    
#     #folder_masked = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/single_blob_center_masked_grid_0"
#     folder_masked = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/single_blob_npt_center_masked_grid_45"
    
#     X_masked = load_blob_dataset(folder_masked)

#     modes, S, mean = compute_pod(X, n_components=20)
#     show_modes(modes)
#     plot_pod_energy_log_scale(S, max_components=20)

#     # test_img = X[0]
#     # recon_img, coeffs = reconstruct_image(test_img, modes, mean)
    
#     for i in range(5):
#         masked_img = X_masked[i]
#         mask = masked_img != 0  # Seuil pour définir les zones non masquées
    
#         recon_img, _ = reconstruct_missing_only(masked_img, mask, modes, mean)
    
#         plt.figure(figsize=(10, 4))
#         plt.subplot(1, 2, 1)
#         plt.imshow(masked_img, cmap="gray", vmin=0, vmax=255)
#         plt.title(f"Image masquée #{i}")
#         plt.axis("off")
    
#         plt.subplot(1, 2, 2)
#         plt.imshow(recon_img, cmap="gray", vmin=0, vmax=255)
#         plt.title("Reconstruction POD")
#         plt.axis("off")
    
#         plt.tight_layout()
#         plt.show()

# if __name__ == "__main__":
#     # 1. Charge l'ensemble d'entraînement pour la POD
#     folder_train = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/data/train/complete"
#     X_train = load_blob_dataset(folder_train)

#     # 2. Calcule la base POD sur le train
#     modes, S, mean = compute_pod(X_train, n_components=20)
#     show_modes(modes)
#     plot_pod_energy_log_scale(S, max_components=20)

#     # 3. Charge les paires test (original, masquée)
#     folder_test_complete = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/data/test/complete"
#     folder_test_masked = "/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Codes/03_PCA/data_generation/data/test/masked"
#     pairs = load_test_pairs(folder_test_complete, folder_test_masked)

#     # 4. Boucle d'évaluation reconstruction et mesure
#     all_metrics = []
    
#     lmbda = 0.001
    
#     for i, (original_img, masked_img) in enumerate(pairs):
#         mask = masked_img != 0  # 1: pixels observés, 0: masqués
#         recon_img, _ = reconstruct_missing_only(masked_img, mask, modes, mean, lmbda=lmbda)

#         # Calcul des métriques (full + masque)
#         metrics = compute_metrics(original_img, recon_img, mask, display=True)
#         metrics["index"] = i
#         all_metrics.append(metrics)

#         # Affichage (optionnel)
#         plt.figure(figsize=(14, 4))
#         plt.subplot(1, 3, 1)
#         plt.imshow(original_img, cmap="gray", vmin=0, vmax=255)
#         plt.title(f"GT #{i}")
#         plt.axis("off")

#         plt.subplot(1, 3, 2)
#         plt.imshow(masked_img, cmap="gray", vmin=0, vmax=255)
#         plt.title("Masqué")
#         plt.axis("off")

#         plt.subplot(1, 3, 3)
#         plt.imshow(recon_img, cmap="gray", vmin=0, vmax=255)
#         plt.title("Reconstruit")
#         plt.axis("off")

#         plt.tight_layout()
#         plt.show()

#     # 5. (Optionnel) Sauvegarder les scores en CSV
#     import pandas as pd
#     df = pd.DataFrame(all_metrics)
#     df.to_csv("scores_reconstruction.csv", index=False)
#     print("Résultats sauvegardés dans scores_reconstruction.csv")
    

#     metrics = [
#     "psnr_full", "ssim_full", "mse_full",
#     "psnr_masked", "ssim_masked", "mse_masked"]
#     for metric in metrics:
#         m = df[metric].mean()
#         s = df[metric].std()
#         print(f"{metric:12s}: {m:.3f} ± {s:.3f}")
    
if __name__ == "__main__":
    # 1. Charge l'ensemble d'entraînement pour la POD
    folder_train = '/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Figure/blob_centre_rotate'
    X_train = load_blob_dataset(folder_train)

    # 2. Calcule la base POD sur le train
    modes, S, mean = compute_pod(X_train, n_components=5)
    show_modes(modes)
    plot_pod_energy_log_scale(S, max_components=20)

    # 3. Charge les images de test à compléter
    folder_test = '/Users/eliotmorard/Desktop/Bureau - MacBook Air de Eliot/Travail/M1 - ENS/TER/Figure/partial_centrate_rotate'
    X_test = load_blob_dataset(folder_test)
    
    

    for i, masked_img in enumerate(X_test):
        # Ici, il faut définir ce qui est masqué (par exemple, pixels à 0 ou à 255…)
        quantile = 0.30  # 40%
        seuil = np.quantile(masked_img, quantile)
        mask = masked_img > seuil 
        
        # mask = masked_img != 0 

        # Reconstruction POD (option régularisation possible)
        recon_img, _ = reconstruct_missing_only(masked_img, mask, modes, mean, lmbda=0.001)
        
        

        # Affichage comparatif
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(masked_img, cmap="gray", vmin=0, vmax=255)
        plt.title("Image à compléter")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(recon_img, cmap="gray", vmin=0, vmax=255)
        plt.title("Reconstruction POD")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
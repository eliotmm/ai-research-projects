#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 14:09:07 2025

@author: eliotmorard
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:26:03 2025

@author: eliotmorard
"""

################################################## UNIFORME
import numpy as np
from scipy.ndimage import gaussian_filter
import imageio.v2 as imageio
import os
import uuid



### --- Pipeline génération image --- ###
def generate_blank_image(heigth : int, width : int) -> np.ndarray:
    """ Génération d'une image vide """
    return np.zeros((heigth, width), dtype=np.uint8)

def add_1_center_blob(image: np.ndarray) -> np.ndarray:
    """Ajoute un blob elliptique gaussien centré dans l'image"""
    a =10
    h, w = image.shape
    center_y, center_x = h // 2 + np.random.uniform(-a, a) , w // 2 + np.random.uniform(-a, a)
    

    sigma_x = np.random.uniform(6, 12)
    sigma_y = np.random.uniform(2, 6)

    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # Blob elliptique centré
    blob = 255 * np.exp(-(((x - center_x) ** 2) / (2 * sigma_x ** 2) +
                          ((y - center_y) ** 2) / (2 * sigma_y ** 2)))

    # Fusion dans l'image
    image = np.maximum(image, blob.astype(np.uint8))
    return image
        
def apply_gaussian_blur(image: np.ndarray, sigma : float) -> np.ndarray:
    blurred = gaussian_filter(image, sigma=sigma)
    blurred = np.clip(blurred, 0, 255)
    return blurred.astype(np.uint8)

def add_mask(image: np.ndarray, mode='grid', value=0, angle=45, thickness=3) -> np.ndarray:
    """
    Ajoute un masque dans l’image (valeur mise à `value`).
    - 'grid' : grille régulière
    - 'stripe' : bande horizontale
    - 'diagonal' : bande diagonale de largeur `thickness`, orientée selon `angle` (en degrés)
    """
    h, w = image.shape
    masked = image.copy()

    if mode == 'grid':
        step = h // 2
        for y in range(0, h, step):
            masked[y:y+thickness, :] = value
        for x in range(0, w, step):
            masked[:, x:x+thickness] = value

    elif mode == 'stripe':
        masked[h//3:2*h//3, :] = value

    elif mode == 'diagonal':
        theta = np.deg2rad(angle)
        cx, cy = w // 2, h // 2

        # Créer une grille d’indices
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

        # Coordonnées centrées
        x_shifted = x - cx
        y_shifted = y - cy

        # Distance au centre projetée perpendiculairement à la direction
        # cos(θ)·dy - sin(θ)·dx donne la distance à la ligne diagonale
        distance = np.abs(np.cos(theta) * y_shifted - np.sin(theta) * x_shifted)

        # Masquer les pixels proches de la ligne
        masked[distance <= thickness / 2] = value

    return masked

# def save_image(image: np.ndarray, path : str) -> str:
#     """Enregistre l'image dans un fichier .png"""
#     # Crée le dossier s'il n'existe pas
#     os.makedirs(folder, exist_ok=True)
#     filename = f"{uuid.uuid4().hex}.png"  # nom unique
#     path = os.path.join(folder, filename)
#     imageio.imwrite(path, image)
#     return path

def save_image_with_folder(image: np.ndarray, folder: str, filename: str) -> str:
    """Enregistre l'image dans un dossier donné avec le nom donné."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    imageio.imwrite(path, image)
    return path

def generate_and_save_train_image(heigth, width, sigma, folder_complete):
    image = generate_blank_image(heigth, width)
    image += 25
    image = add_1_center_blob(image)
    image = apply_gaussian_blur(image, sigma)
    filename = f"{uuid.uuid4().hex}.png"
    save_image_with_folder(image, folder_complete, filename)
    return filename

# if __name__ == "__main__" :
#     n = 50
#     heigth, width = 127, 127
#     sigma = 3.0
#     folder = "single_blob_npt_center_masked_grid_45"
    
#     for k in range (n):
#         image = generate_blank_image(heigth, width)
#         image += 25
#         image = add_1_center_blob(image)
#         image = apply_gaussian_blur(image, sigma)
#         image = add_mask(image, mode='diagonal', angle=45, thickness=5)
#         saved_path = save_image(image, folder)
#         print(f"Image enregistrée : {saved_path}")

if __name__ == "__main__":
    # ---- Génération ENSEMBLE D'ENTRAÎNEMENT (complet seulement) ----
    n_train = 80
    folder_train_complete = "data/train/complete"
    heigth, width = 127, 127
    sigma = 3.0

    for k in range(n_train):
        generate_and_save_train_image(
            heigth, width, sigma,
            folder_complete=folder_train_complete
        )
    print("Train set complet généré.")

    # ---- Génération ENSEMBLE DE TEST (paire complet/masqué) ----
    n_test = 20
    folder_test_complete = "data/test/complete"
    folder_test_masked = "data/test/masked"
    mask_mode = 'diagonal'  # ou 'grid', etc.

    for k in range(n_test):
        image = generate_blank_image(heigth, width)
        image += 25
        image = add_1_center_blob(image)
        image = apply_gaussian_blur(image, sigma)
        filename = f"{uuid.uuid4().hex}.png"
        save_image_with_folder(image, folder_test_complete, filename)
        masked_image = add_mask(image, mode=mask_mode, angle=45, thickness=5)
        save_image_with_folder(masked_image, folder_test_masked, filename)
        print(f"Paire test enregistrée : {filename}")

    print("Test set complet+masqué généré.")
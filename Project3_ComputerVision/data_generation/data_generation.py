#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 13:26:03 2025

@author: eliotmorard
"""

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
    h, w = image.shape
    center_y, center_x = h // 2, w // 2

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

def save_image(image: np.ndarray, path : str) -> str:
    """Enregistre l'image dans un fichier .png"""
    # Crée le dossier s'il n'existe pas
    os.makedirs(folder, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"  # nom unique
    path = os.path.join(folder, filename)
    imageio.imwrite(path, image)
    return path

if __name__ == "__main__" :
    n = 50
    heigth, width = 127, 127
    sigma = 3.0
    folder = "single_blob_center"
    
    for k in range (n):
        image = generate_blank_image(heigth, width)
        image += 25
        image = add_1_center_blob(image)
        image = apply_gaussian_blur(image, sigma)
        saved_path = save_image(image, folder)
        print(f"Image enregistrée : {saved_path}")
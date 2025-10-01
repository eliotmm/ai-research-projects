#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:51:35 2025

@author: eliotmorard
"""

# data.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def create_data_with_labels(dataset_dir, test_size=0.2, random_state=42):
    """
    Parcourt le dossier dataset_dir, qui contient plusieurs sous-dossiers
    (un par classe). Retourne X_train, y_train, X_test, y_test, class_names.
    Les images sont redimensionnées et normalisées.
    """
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
    ])
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    images = []
    labels = []

    for cls_name in class_names:
        cls_dir = os.path.join(dataset_dir, cls_name)
        for file_name in os.listdir(cls_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(cls_dir, file_name)
                img = cv2.imread(file_path)
                if img is None:
                    print(f"[AVERTISSEMENT] Impossible de lire {file_path}")
                    continue
                # Convertir de BGR vers RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # ------------------ ICI, on redimensionne ------------------
                # Par exemple, on veut 224 x 224
                new_w, new_h = 96, 96
                img = cv2.resize(img, (new_w, new_h))  # (width, height)
                
                # -----------------------------------------------------------
                
                # Normalisation [0,1]
                img = img.astype(np.float32) / 255.0

                images.append(img)
                labels.append(class_to_id[cls_name])

    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    # Split train/test ou train/val
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return (X_train, y_train, X_test, y_test, class_names)
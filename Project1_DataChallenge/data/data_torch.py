#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:29:01 2025

@author: eliotmorard
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split

class ImageFolderCustom(Dataset):
    """
    Dataset simple : prend une liste de (image_path, label),
    applique transform, et renvoie (tensor_image, label).
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        # Ouvrir l'image
        image = Image.open(img_path).convert("RGB")
        # Appliquer la transform
        if self.transform:
            image = self.transform(image)
        return image, label

def create_data_with_labels(dataset_dir, test_size=0.2, random_state=42, batch_size=16, subset_size=None):
    """
    Parcourt dataset_dir, qui contient des sous-dossiers "Class1", "Class2", etc.
    Renvoie:
      - train_loader, test_loader (DataLoader PyTorch)
      - class_names (liste)
    """
    class_names = sorted([d for d in os.listdir(dataset_dir) 
                          if os.path.isdir(os.path.join(dataset_dir, d))])

    image_paths = []
    labels = []

    # Parcourt chaque classe
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        # Récupère toutes les images du sous-dossier
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(class_dir, fname)
                image_paths.append(path)
                labels.append(label_idx)

    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, 
        test_size=test_size, 
        stratify=labels, 
        random_state=random_state
    )
    
    # Sous-échantillonage si nécessaire
    if subset_size is not None and subset_size > 0:
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]
        X_test = X_test[:subset_size]
        y_test = y_test[:subset_size]

    # Resize, normalisation
    train_transform = T.Compose([
        T.Resize((224, 224)),  
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],  # Stats ImageNet
                    std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

    # Création des Dataset
    train_dataset = ImageFolderCustom(X_train, y_train, transform=train_transform)
    test_dataset = ImageFolderCustom(X_test, y_test, transform=test_transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, class_names
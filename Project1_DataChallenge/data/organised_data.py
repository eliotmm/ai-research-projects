#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:26:21 2025

@author: eliotmorard
"""

""" 
Création des dossiers avec les différentes classes 

    GOOD (Label = 0) 1235
    Boucle plate (Label = 1) 71
    Lift-off blanc (Label = 2) 270
    Lift-off noir (Label = 3) 104
    Missing (Label = 4) 6472
    Short circuit MOS (Label = 5) 126
    
"""
import csv
import os
import shutil

# Chemin vers le fichier .csv
CSV_FILE = '/Users/eliotmorard/Desktop/Valeo challenge - v1/raw_data_train/Y_train.csv'

# Chemin vers le dossier avec le fichier contenant les images .png
SOURCE_FOLDER = '/Users/eliotmorard/Desktop/Valeo challenge - v1/raw_data_train/input_train'

# Dossier de destination où seront créés les sous-dossiers
DEST_FOLDER = '/Users/eliotmorard/Desktop/Valeo challenge - v1/organised_data_train'


if __name__ == "__main__" :

    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        
        for row in reader:
            image_name = row['filename']      # Nom du fichier image
            label = row['Label']        # Clé ou catégorie
            
            # Chemin du sous-dossier correspondant à la catégorie
            target_folder = os.path.join(DEST_FOLDER, label)
            
            # Crée le sous-dossier s'il n'existe pas déjà
            os.makedirs(target_folder, exist_ok=True)
            
            # Construction du chemin complet de l'image source
            source_image_path = os.path.join(SOURCE_FOLDER, image_name)
            
            # Construction du chemin de destination pour l'image
            dest_image_path = os.path.join(target_folder, image_name)
            
            # Copie l'image
            if os.path.exists(source_image_path):
                shutil.copy2(source_image_path, dest_image_path)
            else:
                print(f"Le fichier {source_image_path} n'existe pas.")

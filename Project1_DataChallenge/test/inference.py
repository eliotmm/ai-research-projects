#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 23:29:38 2025

@author: eliotmorard
"""

import os
import csv
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image

# === CONFIGURATION ===
CSV_FILE = '/Users/eliotmorard/Desktop/Travail/IA/UE Data challenge/Valeo challenge - v1/Supp_files/win_and_lib.csv'
SOURCE_FOLDER = "/Users/eliotmorard/Desktop/Travail/IA/UE Data challenge/Valeo challenge - v1/input_test"
DEST_FOLDER = '/Users/eliotmorard/Desktop/Travail/IA/UE Data challenge/Valeo challenge - v1/output_test'
MODEL_PATH = "/Users/eliotmorard/Desktop/Travail/IA/UE Data challenge/Valeo challenge - v1/code/trained_model.h5"
SUBMISSION_FILE = "/Users/eliotmorard/Desktop/Valeo challenge - v1/submission_predicted.csv"

# === Dictionnaire de rotation/crop ===
rot_crop_data = {
    "Die01": [55,   (340, 120, 500, 680)],
    "Die02": [-44,  (480, 210, 640, 930)],
    "Die03": [134,  (460, 200, 620, 920)],
    "Die04": [35,   (310, 130, 470, 690)]
}

# === Correspondance classe / numéro ===
class_mapping = {
    0: 1,
    1: 0,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6
}

class_mapping = {key: 4 for key in class_mapping}


def rotate_and_crop_image(input_path, output_path, angle, crop_box):
    """Ouvre l'image source, applique rotation + crop, puis enregistre au chemin output_path."""
    with Image.open(input_path) as im:
        rotated = im.rotate(angle, expand=True)  # Rotation + agrandissement du canvas
        cropped = rotated.crop(crop_box)
        cropped.save(output_path, format='PNG')
        
        
        
# === Vérification du format du CSV ===
df = pd.read_csv(CSV_FILE)
print("Aperçu du fichier CSV original :")
print(df.head())

with open(CSV_FILE, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    if ';' in first_line:
        print("⚠️ Le fichier CSV utilise ';' comme séparateur.")
        df = pd.read_csv(CSV_FILE, delimiter=';')

# === Assurer la création du dossier de destination ===
os.makedirs(DEST_FOLDER, exist_ok=True)

# === Rotation et crop des images ===
processed_images = []

for _, row in df.iterrows():
    image_name = row['filename']
    lib = row['lib']

    source_image_path = os.path.join(SOURCE_FOLDER, image_name)
    dest_image_path = os.path.join(DEST_FOLDER, image_name)

    if not os.path.exists(source_image_path):
        print(f"Image non trouvée : {source_image_path}")
        continue

    if lib in rot_crop_data:
        angle, crop_box = rot_crop_data[lib]
        rotate_and_crop_image(source_image_path, dest_image_path, angle, crop_box)
    else:
        shutil.copy2(source_image_path, dest_image_path)

    processed_images.append(image_name)

print(f"{len(processed_images)} images prétraitées et stockées dans {DEST_FOLDER}")

# === CHARGEMENT DU MODÈLE ===
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(img_path, target_size=(96, 96)):
    """Charge et prétraite une image pour la classification"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# === PRÉDICTION DES IMAGES ===
predicted_labels = []
for img_name in processed_images:
    img_path = os.path.join(DEST_FOLDER, img_name)
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    predicted_labels.append(class_mapping[predicted_label])

print(f"Prédictions effectuées sur {len(predicted_labels)} images")

# === Génération du fichier de soumission ===
df["Label"] = predicted_labels
df["Label"] = df["Label"].astype(int)
print("Aperçu du fichier final avant export :")
print(df.head())

#df.reset_index(drop=True, inplace=True)
df.to_csv(SUBMISSION_FILE, index=True)
print(f"Fichier de soumission généré : {SUBMISSION_FILE}")


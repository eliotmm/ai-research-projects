import csv
import os
import shutil
from PIL import Image

# Chemin vers le fichier .csv
CSV_FILE = '/Users/eliotmorard/Desktop/Valeo challenge - v1/raw_data_train/Y_train.csv'

# Chemin vers le dossier avec les images source (.png)
SOURCE_FOLDER = '/Users/eliotmorard/Desktop/Valeo challenge - v1/raw_data_train/input_train'

# Dossier de destination (sous-dossiers créés, images copiées ou transformées)
DEST_FOLDER = '/Users/eliotmorard/Desktop/Valeo challenge - v1/rotate_data_train'

# Dictionnaire rotation/crop (exemple)
rot_crop_data = {
    "Die01": [55,   (340, 120, 500, 680)],  # angle : (left, upper, right, lower)
    "Die02": [-44,  (480, 210, 640, 930)],
    "Die03": [134,  (460, 200, 620, 920)],
    "Die04": [35,   (310, 130, 470, 690)]
}

def rotate_and_crop_image(input_path, output_path, angle, crop_box):
    """Ouvre l'image source, applique rotation + crop, puis enregistre au chemin output_path."""
    with Image.open(input_path) as im:
        rotated = im.rotate(angle, expand=True)  # Rotation + agrandissement du canvas
        cropped = rotated.crop(crop_box)
        cropped.save(output_path, format='PNG')

if __name__ == "__main__":
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')  # Ajuster si séparateur ';'
        
        for row in reader:
            # Récupération des 3 infos
            image_name = row['filename']  # Nom du fichier .png
            label = row['Label']          # Sous-dossier à créer
            lib = row['lib']              # rotation/crop
            
            # Dossier cible 
            target_folder = os.path.join(DEST_FOLDER, label)
            os.makedirs(target_folder, exist_ok=True)
            
            # Chemins complet source/destination
            source_image_path = os.path.join(SOURCE_FOLDER, image_name)
            dest_image_path = os.path.join(target_folder, image_name)
            
            # Vérification que l'image existe
            if not os.path.exists(source_image_path):
                print(f"Le fichier {source_image_path} n'existe pas.")
                continue
            
            # Si lib est dans rot_crop_data, on applique rotation + crop
            if lib in rot_crop_data:
                angle, crop_box = rot_crop_data[lib]
                rotate_and_crop_image(source_image_path, dest_image_path, angle, crop_box)
            else:
                # Sinon, on copie telle quelle
                shutil.copy2(source_image_path, dest_image_path)
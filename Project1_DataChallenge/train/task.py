#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:52:26 2025

@author: eliotmorard
"""

# task.py


import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

import data
import model


cost_matrix_evaluate = np.array([
        [0, 10_000, 1, 1, 1, 1],
        [100, 0, 100, 100, 100, 100],
        [1, 10_000, 0, 1, 1, 1],
        [1, 10_000, 1, 0, 1, 1],
        [1, 10_000, 1, 1, 0, 1],
        [1, 10_000, 1, 1, 1, 0]
    ])

def compute_pwa(conf_matrix, cost_matrix):
    """
    Calcule le Penalty Weighted Accuracy (PWA)
    
    Paramètres :
    - conf_matrix : np.array de shape (N, N), matrice de confusion
    - cost_matrix : np.array de shape (N, N), matrice de coût
    
    Retourne :
    - PWA (float)
    """
    conf_matrix = np.array(conf_matrix)
    cost_matrix = np.array(cost_matrix)
    
    # Vérifions que les matrices ont bien la même taille
    assert conf_matrix.shape == cost_matrix.shape, "Les matrices doivent avoir la même taille !"
    
    # Calcul du total des pénalités
    total_penalty = np.sum(conf_matrix * cost_matrix)

    # Calcul du worst-case penalty (hypothèse : chaque vrai échantillon est classé dans la pire catégorie)
    worst_case_penalty = 0
    total_samples_per_class = np.sum(conf_matrix, axis=1)  # Nombre total de samples pour chaque classe réelle
    
    for i in range(len(cost_matrix)):  # Parcours des classes réelles
        worst_case_penalty += total_samples_per_class[i] * np.max(cost_matrix[i])

    # Calcul du PWA
    pwa = 1 - (total_penalty / worst_case_penalty)
    
    return pwa


def visualize_tsne(cnn_model, X, y, class_names):
    """
    Fait un t-SNE sur les embeddings du modèle pour X
    et affiche un scatter plot en 2D, coloré par la classe y.
    """
    # 1) Construire un sous-modèle pour extraire les features avant la dernière Dense
    # On repère la couche dense (128) si c’est la 2e couche en partant de la fin
    # Sinon, on peut nommer la couche ou la chercher par index.
    feature_extractor = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[-2].output  # La couche Dense(128) avant le Dense final
    )
    
    # 2) Extraire les features
    features = feature_extractor.predict(X)  # shape (N, 128)
    
    # 3) Appliquer t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # 4) Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=y, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.title("Visualisation t-SNE des embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

def train_and_evaluate(dataset_dir):
    # 1) Chargement des données (X_train, y_train, X_test, y_test)
    X_train, y_train, X_test, y_test, class_names = data.create_data_with_labels(dataset_dir,
                                                                                test_size=0.2)
    #### Réduction pour accéleration
    
    # subset_size = 300  # ou un autre nombre réduit
    # X_train = X_train[:subset_size]
    # y_train = y_train[:subset_size]
    # X_test = X_test[:subset_size]
    # y_test = y_test[:subset_size]
    
    # 2) Construire un input_layer adapté
    # On suppose X_train a shape (None, 160, 560, 3)
    input_shape = X_train.shape[1:]  # (160, 560, 3)
    input_layer = tf.keras.layers.Input(shape=input_shape, name='input_image')

    # 3) Récupérer le nombre de classes
    num_classes = len(class_names)

    # 4) Obtenir le modèle
    cnn_model = model.solution(num_classes=num_classes)

    # 5) Entraîner
    history = cnn_model.fit(
        X_train, y_train,
        validation_split=0.2,  # fraction de train pour la validation
        epochs=model.get_epochs(),
        batch_size=model.get_batch_size(),
        shuffle=True
    )
    
    # Sauvegarde du modèle après entraînement
    model_path = "trained_model.h5"
    cnn_model.save(model_path)
    print(f"\n✅ Modèle sauvegardé dans {model_path}")
    
    # === Plot des courbes d'entraînement ===
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Courbe de perte')
    plt.show()
    
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Courbe d’accuracy')
    plt.show()
    
    # 6) Évaluer sur le test set
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=1)
    print(f"\nTest accuracy = {test_acc:.4f}")

    # 7) Matrice de confusion
    predictions = cnn_model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(y_test, predicted_labels)
    print("Matrice de confusion :\n", cm)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matrice de confusion")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.show()
    
    # 8) Visualiser l’espace d’embeddings par t-SNE
    visualize_tsne(cnn_model, X_test, y_test, class_names)
    
    # 9) report
    print(classification_report(
    y_test, 
    predicted_labels, 
    target_names=class_names, 
    zero_division=0  # Au lieu de lever un warning, ça force la valeur à 0.
    ))
    
    # Calcul du PWA
    pwa_result = compute_pwa(cm, cost_matrix_evaluate)
    print(f"\n✅ Score PWA final : {pwa_result:.4f}")
        
    

if __name__ == "__main__":
    dataset_dir = "/Users/eliotmorard/Desktop/Travail/IA/UE Data challenge/Valeo challenge - v1/organised_data_train"  # Dossier avec sous-dossiers => /dataset/Class1, /dataset/Class2, ...
    train_and_evaluate(dataset_dir)
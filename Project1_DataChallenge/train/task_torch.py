#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:29:52 2025

@author: eliotmorard
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

from data_torch import create_data_with_labels
from model_torch import create_deit_model, get_batch_size, get_epochs
from tqdm import tqdm 
# ---------------------------
# Optionnel : t-SNE
# ---------------------------


import torch.nn.functional as F

MODEL_PATH = "/Users/eliotmorard/Desktop/Travail/IA/UE Data challenge/Valeo challenge - v1/code/trained_model.pth"


C = 6  # 6 classes
# cost_matrix_values = [
#   # y=0 => Boucle plate
#   [0.0,    100.0, 1.0,    1.0, 1.0, 1.0],
#   # y=1 => GOOD
#   [10.0,  0,       10.0,  10.0, 10.0, 10.0],
#   # y=2 => Lift-off blanc
#   [1.0,    100.0, 0.0,    1.0, 1.0, 1.0],
#   # y=3 => Lift-off noir
#   [1.0,    100.0, 1.0,    0.0, 1.0, 1.0],
#   # y=4 => Missing
#   [1.0,    100.0, 1.0,    1.0, 0.0, 1.0],
#   # y=5 => Short circuit MOS
#   [1.0,    100.0, 1.0,    1.0, 1.0, 0.0]
# ]

cost_matrix_values = [
  [0.0,  10.0,  1.0,   1.0,  1.0,  1.0],  # Boucle plate
  [3.0,  0.0,  3.0, 3.0, 3.0, 3.0],       # GOOD
  [1.0,  10.0,  0.0,   1.0,  1.0,  1.0],  # Lift-off blanc
  [1.0,  10.0,  1.0,   0.0,  1.0,  1.0],  # Lift-off noir
  [1.0,  10.0,  1.0,   1.0,  0.0,  1.0],  # Missing
  [1.0,  10.0,  1.0,   1.0,  1.0,  0.0]   # Short circuit MOS
]

cost_matrix = torch.tensor(cost_matrix_values)

cost_matrix = torch.tensor(cost_matrix_values).to(torch.device("mps" if torch.backends.mps.is_available() else "cpu"))


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
    
    # Vérification que les matrices ont bien la même taille
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

def cost_matrix_loss(logits, labels, cost_matrix):
    device = logits.device  # Récupérer le device du modèle (MPS ou CPU)
    
    probs = F.softmax(logits, dim=1)  # [B, C]

    # Utilisation GPU
    cost_matrix = cost_matrix.to(device)  
    labels = labels.to(device)

    # Indexation correcte
    cost_rows = torch.index_select(cost_matrix, 0, labels)  # [B, C]

    # Calcul du coût pondéré
    cost_per_sample = torch.sum(cost_rows * probs, dim=1)  # [B]
    return torch.mean(cost_per_sample)  # Moyenne sur le batch

def extract_features(model, dataloader, device):
    """
    Parcourt le dataloader et récupère les features AVANT la couche de classification finale.
    Suppose qu'on peut accéder à model.get_layer ou similaire.
    
    Pour DeiT sous timm, on peut soit :
    - ajuster le forward du modèle pour couper avant la 'head'
    - ou cloner le modèle et remplacer la head par une identity
    - ou modifier la forward du timm model
    
    Ici, on va opter pour la 2e solution (monter un sous-modèle) 
    => On peut faire un hack : model.forward_features
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, memory_format=torch.channels_last)
            
            # "forward_features" renvoie les features avant la classification
            # pour la plupart des modèles timm, dont DeiT
            feats = model.forward_features(images)  # shape [batch, embed_dim]
            feats = feats[:, 0, :]
            
            all_features.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_features = np.concatenate(all_features, axis=0)  # [N, embed_dim]
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels


def visualize_tsne(model, dataloader, class_names, device):
    """
    Calcule un t-SNE sur les features extraites du modèle.
    """
    features, labels = extract_features(model, dataloader, device)
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    features_2d = tsne.fit_transform(features)  # [N, 2]
    
    for idx, name in enumerate(class_names):
        print(f"{idx} -> {name}")
    
    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(class_names)))
    plt.title("Visualisation t-SNE des embeddings (DeiT)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

# ---------------------------
# Entraînement & évaluation
# ---------------------------

def train_and_evaluate(dataset_dir):
    # 1) Charger data
    batch_size = get_batch_size()
    train_loader, test_loader, class_names = create_data_with_labels(
        dataset_dir,
        test_size=0.2,
        random_state=42,
        batch_size=batch_size,
        subset_size=None
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")  # Vérifie que c'est bien "mps"
    
    # 2) Créer le modèle
    model = create_deit_model(num_classes=len(class_names), device=device)
    
    # Optimisation MPS : Passage au format channels_last pour meilleure perf GPU
    model = model.to(memory_format=torch.channels_last)
    
    # 3) Définir l'optimiseur et la loss
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = lambda logits, labels: cost_matrix_loss(logits, labels, cost_matrix)
    #criterion = nn.CrossEntropyLoss()
    epochs = get_epochs()
    
    # 4) Boucle d'entraînement
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        # Ajout de tqdm pour afficher chaque batch
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            labels = labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(images)
            loss = cost_matrix_loss(outputs, labels, cost_matrix)
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Mise à jour dynamique de la progress bar
            loop.set_postfix(loss=loss.item(), acc=correct/total)
        
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # --- Phase validation ---
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}")
    
   
    
    
    # Sauvegarde des poids du modèle PyTorch
    torch.save(model.state_dict(), MODEL_PATH)
    
    print(f"Modèle sauvegardé sous {MODEL_PATH}")
    
    # 5) Tracer les courbes d'entraînement
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Courbe de perte (PyTorch + DeiT)')
    plt.show()

    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Courbe d’accuracy (PyTorch + DeiT)')
    plt.show()
    
    # 6) Évaluation finale (test set)
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 7) Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)
    print("Matrice de confusion :\n", cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matrice de confusion (DeiT)")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.show()
    
    # 8) Classification report
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # 9) Visualisation t-SNE
    visualize_tsne(model, test_loader, class_names, device)
    
    # 10) PWA
    pwa_score = compute_pwa(cm, cost_matrix_evaluate)
    print(f"Penalty Weighted Accuracy (PWA) : {pwa_score:.4f}")


if __name__ == "__main__":
    # Lancer le training
    dataset_dir = "/Users/eliotmorard/Desktop/Travail/IA/UE Data challenge/Valeo challenge - v1/organised_data_train"
    train_and_evaluate(dataset_dir)
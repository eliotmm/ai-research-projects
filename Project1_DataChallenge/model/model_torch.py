#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 10:29:30 2025

@author: eliotmorard
"""


import timm
import torch
import torch.nn as nn

def get_batch_size():
    return 64

def get_epochs():
    return 15

def create_deit_model(num_classes=6, model_name='deit_tiny_patch16_224', pretrained=True, device="cpu"):
    """
    Charge un DeiT depuis timm, remplace la dernière couche par num_classes.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    ).to(device)
    
    # 🔥 Geler toutes les couches du backbone
    for param in model.parameters():
        param.requires_grad = False

    # 🔥 Dégeler uniquement la tête de classification
    for param in model.head.parameters():
        param.requires_grad = True
        
    # ✅ Déplacer tous les paramètres vers `device` pour éviter les erreurs
    for name, param in model.named_parameters():
        param.data = param.data.to(device)  # 🔥 Déplacement explicite
        print(f"{name}: {param.device}")  # Vérification (doit être `mps:0` partout)
    
    return model
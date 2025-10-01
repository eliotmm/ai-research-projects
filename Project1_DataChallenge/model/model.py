#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:40:14 2025

@author: eliotmorard
"""

# model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet101, EfficientNetV2L, ResNet50, EfficientNetV2M, EfficientNetB0, EfficientNetB5, ResNet50V2, ResNet152V2
from tensorflow.keras.optimizers import SGD, AdamW, RMSprop, Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau



    
def get_batch_size():
    """Batch size pour l'entraînement."""
    return 16

def get_epochs():
    """Nombre d'époques pour l'entraînement."""
    return 10

def cbam_block(input_tensor, reduction_ratio=8):
    """CBAM : Channel + Spatial attention."""
    channel = input_tensor.shape[-1]

    # Channel Attention
    avg_pool = tf.reduce_mean(input_tensor, axis=[1, 2], keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=[1, 2], keepdims=True)
    shared_dense_one = keras.layers.Dense(channel // reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True)
    shared_dense_two = keras.layers.Dense(channel, kernel_initializer='he_normal', use_bias=True)

    avg_out = shared_dense_two(shared_dense_one(avg_pool))
    max_out = shared_dense_two(shared_dense_one(max_pool))
    channel_attention = keras.layers.Activation('sigmoid')(avg_out + max_out)
    channel_refined = keras.layers.Multiply()([input_tensor, channel_attention])

    # Spatial Attention
    avg_pool = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    spatial_attention = keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    refined_output = keras.layers.Multiply()([channel_refined, spatial_attention])

    return refined_output

def solution(num_classes=6):
    """
    Construit et compile un modèle de classification basé sur MobileNetV2.
    - input_layer: un tf.keras.layers.Input(shape=(160, 560, 3)) par exemple
    - num_classes: nombre de classes en sortie
    Retourne un tf.keras.Model compilé.
    """
    input_layer = Input(shape=(96, 96, 3))
    
    # Charger MobileNetV2 pré-entraîné, sans la top layer
    base_model = ResNet152V2(
        weights='imagenet',
        include_top=False,  # on enlève la dernière couche Dense
        input_tensor=input_layer
    )
    # On gèle le backbone
    base_model.trainable = False  # freeze
    for layer in base_model.layers[-24:]:
        layer.trainable = True
        
    # 👉 On ajoute le CBAM après le backbone


    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(num_classes, activation='softmax')(x)

    
    model = Model(inputs=input_layer, outputs=output, name="transfer_mobilenetv2")
    
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-5)
    
    # Compilation
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

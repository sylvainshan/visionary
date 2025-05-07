import argparse
import time
import torch
import pickle
import h5py
import os
import re
import csv
import glob
import gdown
import random
import gc
import copy

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from scipy.stats import pearsonr, rankdata, spearmanr

from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.models.feature_extraction import create_feature_extractor

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# =============================================================================
# UTILS : Part 1 - Linear and Task-Driven Models
# =============================================================================

def compute_pca(X_train, X_val, n_components):
    """Perform PCA on the training set, transform both training and validation sets, and return the transformed data."
    
    Args:
        X_train (numpy.ndarray): Training features.
        X_val (numpy.ndarray): Validation features.
        n_components (int): Number of principal components to keep.

    Returns:
        tuple: Transformed training and validation sets.
    """
    print("Computing PCA... ", end='', flush=True)
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)

    print(f"done.")
    return X_train, X_val 

def get_pca(X_train, X_val, n_components, model_type='linear_model'):

    # Define the pickle file path
    pkl_file = f'out/linear_models/pca_{model_type}_{n_components}pcs.pkl'
    os.makedirs("out/linear_models", exist_ok=True)

    # Check if the pickle file exists
    if os.path.exists(pkl_file):
        print("PCA Pickle file found, variables loaded.")
        
        # Load the PCA model and transformed data from the pickle file
        with open(pkl_file, 'rb') as f:
            pca_data = pickle.load(f)
        
        # Extract the PCA model and transformed data
        X_train_pca = pca_data['X_train_pca']
        X_val_pca = pca_data['X_val_pca']

    else:
        print("PCA Pickle file does not exist.")
        
        # Compute PCA if the pickle file does not exist
        X_train_pca, X_val_pca = compute_pca(X_train,X_val,n_components)
        
        # Save the PCA model and transformed data as a pickle file
        with open(pkl_file, 'wb') as f:
            pickle.dump({'X_train_pca': X_train_pca, 'X_val_pca': X_val_pca}, f)
        
        print("PCA model and transformed data saved as pickle!")

    return X_train_pca, X_val_pca


# =============================================================================
# UTILS : Part 2 - Shallow CNN Data-Driven
# =============================================================================

class ITDataset(Dataset):
    def __init__(self, stimuli, neural_responses, objects_cat,transform=None):
        self.stimuli = stimuli
        self.neural_responses = neural_responses
        self.objects_cat = objects_cat
        self.transform = transform

    def __len__(self):
        return self.stimuli.shape[0]

    def __getitem__(self, idx):
        img_np = self.stimuli[idx]
        label = self.neural_responses[idx]
        object_cat = self.objects_cat[idx]
        if self.transform:
            img = np.transpose(img_np, (1,2,0))
            if img.dtype != np.uint8:
                if img.max() <= 1.0 and img.min() >= 0:
                    img = (img * 255).astype(np.uint8)
                else: 
                    img = img.astype(np.uint8)
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        else:
            img = torch.tensor(img_np, dtype=torch.float32)
        return img, torch.tensor(label, dtype=torch.float32), object_cat

# CNN Definition
class ShallowCNN(nn.Module):
    def __init__(self, num_neurons):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(128, num_neurons)

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)

# Training & Evaluation
def train_model(model, train_loader, val_loader, device,
                num_epochs=200, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # decay LR by 10× after epoch 60
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1)
    model.to(device)

    train_losses, val_losses = [], []
    train_evs, val_evs = [], []

    for epoch in range(1, num_epochs+1):
        model.train()
        running_loss = 0.0
        for images, labels, objects_cat in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        # record train loss & EV
        model.eval()
        with torch.no_grad():
            preds_train, targs_train = [], []
            for images, labels, objects_cat in train_loader:
                images = images.to(device)
                out = model(images).cpu().numpy()
                preds_train.append(out)
                targs_train.append(labels.numpy())
            preds_train = np.concatenate(preds_train)
            targs_train = np.concatenate(targs_train)
            train_loss = ((preds_train - targs_train)**2).mean()
            train_ev = explained_variance_score(targs_train, preds_train)

            preds_val, targs_val = [], []
            for images, labels, objects_cat in val_loader:
                images = images.to(device)
                out = model(images).cpu().numpy()
                preds_val.append(out)
                targs_val.append(labels.numpy())
            preds_val = np.concatenate(preds_val)
            targs_val = np.concatenate(targs_val)
            val_loss = ((preds_val - targs_val)**2).mean()
            val_ev = explained_variance_score(targs_val, preds_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_evs.append(train_ev)
        val_evs.append(val_ev)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs} | LR={current_lr:.1e} "
              f"Train Loss={train_loss:.4f} Val Loss={val_loss:.4f} "
              f"Train EV={train_ev:.4f} Val EV={val_ev:.4f}"
              f" Best Val EV={max(val_evs):.4f}")

    return model, {'train_loss': train_losses, 'val_loss': val_losses,
                   'train_ev': train_evs, 'val_ev': val_evs}

def evaluate_model(model, loader, device):
    model.eval()
    preds, targs, objects_cat = [], [], []
    with torch.no_grad():
        for images, labels, obj_cat in loader:
            images = images.to(device)
            preds.append(model(images).cpu().numpy())
            targs.append(labels.numpy())
            objects_cat.append(obj_cat)
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    objects_cat = np.concatenate(objects_cat)
    #print("Final R²:", r2_score(targs, preds))
    #print("Final EV:", explained_variance_score(targs, preds))
    return preds, targs, objects_cat

# Plotting
def save_training_plots(metrics, folder="plots_cnn"):
    os.makedirs(folder, exist_ok=True)
    epochs = np.arange(1, len(metrics['train_loss'])+1)
    plt.figure(); plt.plot(epochs, metrics['train_loss'], label='Train Loss'); plt.plot(epochs, metrics['val_loss'], label='Val Loss'); plt.legend(); plt.savefig(os.path.join(folder,'loss_curve.png')); plt.close()
    plt.figure(); plt.plot(epochs, metrics['train_ev'], label='Train EV'); plt.plot(epochs, metrics['val_ev'], label='Val EV'); plt.legend(); plt.savefig(os.path.join(folder,'ev_curve.png')); plt.close()


# =============================================================================
# UTILS : Part 3 - ResNetAdapter & VGG_BN Architectures definitions, training function
#         (same evaluate_model function as Part 2 is used for both architectures)
# =============================================================================

class ResNetAdapter(nn.Module):
    """
    ResNet model with a specified version (e.g., resnet18, resnet34, resnet50, resnet101, resnet152) and a custom classifier.
    """
    def __init__(self, num_neurons, resnet_version='resnet18', pretrained=True, classifier_config=None, freeze_features=True, unfreeze_blocks=0):
        super().__init__()
        self.resnet_version = resnet_version.lower()

        # --- Load Pretrained ResNet weights if required ---
        if self.resnet_version == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet18(weights=weights)
            num_ftrs = model.fc.in_features # 512
        elif self.resnet_version == 'resnet34':
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet34(weights=weights)
            num_ftrs = model.fc.in_features # 512
        elif self.resnet_version == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None 
            model = resnet50(weights=weights)
            num_ftrs = model.fc.in_features # 2048
        elif self.resnet_version == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet101(weights=weights)
            num_ftrs = model.fc.in_features # 2048
        elif self.resnet_version == 'resnet152':
            weights = ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet152(weights=weights)
            num_ftrs = model.fc.in_features # 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        if pretrained:
            print(f"Loading pretrained weights for {self.resnet_version}...")
        else:
             print(f"Initializing {self.resnet_version} with random weights...")

        # --- Extract Feature Layers ---
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        self.num_feature_outputs = num_ftrs

        # --- If freeze_features is True, pretrained layers will not be modified ---
        if freeze_features:
            print("Freezing ResNet feature layers...")
            for param in self.features.parameters():
                param.requires_grad = False

            # We add the option to unfreeze 'unfreeze_blocks' of the last ResNet blocks
            if unfreeze_blocks > 0:
                resnet_blocks = [self.features[7], self.features[6], self.features[5], self.features[4]] # layer4, layer3, layer2, layer1
                blocks_to_unfreeze = min(unfreeze_blocks, len(resnet_blocks))
                print(f"Unfreezing last {blocks_to_unfreeze} ResNet block(s)...")
                for i in range(blocks_to_unfreeze):
                    for param in resnet_blocks[i].parameters():
                        param.requires_grad = True

        # --- Define Classifier Dynamically ---  
        if classifier_config is None:
            # Default classifier configuration if None provided
            classifier_config = {
                'hidden_sizes': [1024], # Smaller default for ResNet features
                'activation': 'relu',
                'dropout': 0.5
            }

        layers = []
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        input_size = self.num_feature_outputs # Input size is determined by ResNet features

        # Defining the activation function
        activation_name = classifier_config.get('activation', 'relu').lower()
        if activation_name == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_name == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        dropout_rate = classifier_config.get('dropout', 0.5)
        hidden_sizes = classifier_config.get('hidden_sizes', [1024])

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, num_neurons))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class VGG_BN(nn.Module):
    """
    VGG-inspired model with 5 convolutional blocks with batch normalization and a custom classifier. 

    The model can be initialized with pretrained weights from VGG11_BN and allows for dynamic classifier configuration.
    """
    def __init__(self, num_neurons, pretrained=True, classifier_config=None, freeze_features=False, unfreeze_blocks=0): 
        super().__init__()

        # --- Feature Extractor with 5 convolutional layers ---
        self.features = nn.Sequential(  
            # Block 1: 64 filters   
            nn.Conv2d(3, 64, kernel_size=3, padding=1),     # our_conv1 (0)  
            nn.BatchNorm2d(64),                             # our_bn1   (1)
            nn.ReLU(inplace=True),                          # our_relu1 (2)
            nn.MaxPool2d(kernel_size=2, stride=2),          # our_pool1 (3)

            # Block 2: 128 filters  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),   # our_conv2 (4)
            nn.BatchNorm2d(128),                            # our_bn2   (5) 
            nn.ReLU(inplace=True),                          # our_relu2 (6)
            nn.MaxPool2d(kernel_size=2, stride=2),          # our_pool2 (7)

            # Block 3: 256 filters  
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # our_conv3 (8)
            nn.BatchNorm2d(256),                            # our_bn3   (9)
            nn.ReLU(inplace=True),                          # our_relu3 (10)
            nn.MaxPool2d(kernel_size=2, stride=2),          # our_pool3 (11)

            # Block 4: 256 filters  
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # our_conv4 (12)
            nn.BatchNorm2d(256),                            # our_bn4   (13)
            nn.ReLU(inplace=True),                          # our_relu4 (14)
            nn.MaxPool2d(kernel_size=2, stride=2),          # our_pool4 (15)

            # Block 5: 256 filters  
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # our_conv5 (16)
            nn.BatchNorm2d(256),                            # our_bn5   (17)
            nn.ReLU(inplace=True),                          # our_relu5 (18)
            nn.MaxPool2d(kernel_size=2, stride=2),          # our_pool5 (19)
        )

        # --- Pretrained Weights Loading ---
        if pretrained:
            print("Loading pretrained weights from vgg11_bn...")
            try:
                # Load weights from VGG11_BN
                vgg_bn_model = vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
                vgg_layers = list(vgg_bn_model.features.children())
                our_layers = list(self.features.children())

                with torch.no_grad():
                    layers_to_copy = [  
                        (0, 1, 0, 1),     # (our_conv1, our_bn1, vgg_conv1, vgg_bn1)  
                        (4, 5, 4, 5),     # (our_conv2, our_bn2, vgg_conv2, vgg_bn2)
                        (8, 9, 8, 9),     # (our_conv3, our_bn3, vgg_conv3, vgg_bn3)
                        (12, 13, 11, 12), # (our_conv4, our_bn4, vgg_conv4, vgg_bn4)
                        (16, 17, 15, 16)  # (our_conv5, our_bn5, vgg_conv5, vgg_bn5)
                    ]
                    print("Mapping VGG11_BN layers to VGG_BN layers:")
                    for i, (our_c_idx, our_bn_idx, vgg_c_idx, vgg_bn_idx) in enumerate(layers_to_copy):
                         if (our_c_idx < len(our_layers) and vgg_c_idx < len(vgg_layers) and
                             our_bn_idx < len(our_layers) and vgg_bn_idx < len(vgg_layers) and
                             isinstance(our_layers[our_c_idx], nn.Conv2d) and isinstance(vgg_layers[vgg_c_idx], nn.Conv2d) and
                             isinstance(our_layers[our_bn_idx], nn.BatchNorm2d) and isinstance(vgg_layers[vgg_bn_idx], nn.BatchNorm2d)):

                            print(f"  Layer {i+1}: Copying VGG11_BN Conv {vgg_c_idx} -> {our_c_idx} and VGG11_BN BN {vgg_bn_idx} -> {our_bn_idx}")
                            our_conv = our_layers[our_c_idx]
                            vgg_conv = vgg_layers[vgg_c_idx]
                            our_bn = our_layers[our_bn_idx]
                            vgg_bn = vgg_layers[vgg_bn_idx]

                            # Adapt weights for last two convolutional layers
                            if our_c_idx in [12, 16]:
                                print(f"    Adapting Conv weights/bias (VGG out: {vgg_conv.out_channels}, Our out: {our_conv.out_channels})")
                                our_conv.weight.copy_(vgg_conv.weight[:our_conv.out_channels, :vgg_conv.in_channels])
                                our_conv.bias.copy_(vgg_conv.bias[:our_conv.out_channels])
                            else:
                                if our_conv.weight.shape == vgg_conv.weight.shape:
                                    our_conv.weight.copy_(vgg_conv.weight)
                                    our_conv.bias.copy_(vgg_conv.bias)
                                else:
                                     print(f"    Warning: Shape mismatch for Conv layer {i+1}. Skipping weight copy.")
                                     continue # Skip BN copy too if conv failed

                            # Adapt weights for last two BN layers
                            if our_bn_idx in [13, 17]:
                                print(f"    Adapting BN params (VGG features: {vgg_bn.num_features}, Our features: {our_bn.num_features})")
                                num_features_to_copy = min(our_bn.num_features, vgg_bn.num_features)
                                our_bn.weight.copy_(vgg_bn.weight[:num_features_to_copy])
                                our_bn.bias.copy_(vgg_bn.bias[:num_features_to_copy])
                                our_bn.running_mean.copy_(vgg_bn.running_mean[:num_features_to_copy])
                                our_bn.running_var.copy_(vgg_bn.running_var[:num_features_to_copy])
                            else:
                                if our_bn.weight.shape == vgg_bn.weight.shape:
                                    our_bn.weight.copy_(vgg_bn.weight)
                                    our_bn.bias.copy_(vgg_bn.bias)
                                    our_bn.running_mean.copy_(vgg_bn.running_mean)
                                    our_bn.running_var.copy_(vgg_bn.running_var)
                                else:
                                    print(f"    Warning: Shape mismatch for BN layer {i+1}. Skipping BN copy.")
                         else:
                             print(f"  Warning: Index out of bounds or type mismatch for layer mapping {i+1}. Skipping.")

            except Exception as e:
                print(f"Error loading pretrained weights: {e}. Proceeding with random initialization.")

        # --- Freeze Features if Required ---
        # In this case, only the classifier will be trained
        if freeze_features:
            print("Freezing VGG_BN feature layers...")
            for param in self.features.parameters():
                param.requires_grad = False
            
            # If the number 'unfreeze_blocks' is specified, unfreeze the last blocks to allow training
            if unfreeze_blocks > 0:
                layers_per_block = 4 
                total_feature_layers = len(self.features)
                layers_to_unfreeze = min(layers_per_block * unfreeze_blocks, total_feature_layers)
                
                start_index = max(0, total_feature_layers - layers_to_unfreeze)
                print(f"Unfreezing last {unfreeze_blocks} block(s) (layers {start_index} to {total_feature_layers-1})")
                for i in range(start_index, total_feature_layers):
                    for param in self.features[i].parameters():
                        param.requires_grad = True

        # --- Define Classifier Dynamically ---
        if classifier_config is None:
            # Default classifier configuration if None provided
            classifier_config = {
                'hidden_sizes': [2048, 2048], 
                'activation': 'relu', 
                'dropout': 0.27
            }
        
        # The classifier head has the following structure:
        # AdaptiveAvgPool2d -> Flatten -> Classifier Layers 
        layers = []
        input_size = 256 * 7 * 7 # Output size after features and AdaptiveAvgPool2d((7, 7))
        layers.append(nn.AdaptiveAvgPool2d((7, 7)))
        layers.append(nn.Flatten())

        activation_name = classifier_config.get('activation', 'relu').lower()
        if activation_name == 'relu':
            activation_fn = nn.ReLU(inplace=True)
        elif activation_name == 'gelu':
            activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

        dropout_rate = classifier_config.get('dropout', 0.27)
        hidden_sizes = classifier_config.get('hidden_sizes', [2048, 2048])

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(activation_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size

        layers.append(nn.Linear(input_size, num_neurons))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model_best(model, train_loader, val_loader, device, optimizer_config, scheduler_config, num_epochs=50, patience=10):
    """Trains a model with early stopping and displays metrics."""
    # --- Initialize Loss Function ---
    criterion = nn.MSELoss()

    # --- Initialize Optimizer ---
    optimizer_name = optimizer_config.get('name', 'AdamW').lower()
    optimizer_params = optimizer_config.get('params', {'lr': 1e-4, 'weight_decay': 1e-5})
    print(f"Initializing Optimizer: {optimizer_name} with params: {optimizer_params}")
    if optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **optimizer_params)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_name == 'sgd':
         optimizer = optim.SGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # --- Initialize Scheduler ---
    # Default to CosineAnnealingLR if not specified
    scheduler_name = scheduler_config.get('name', 'CosineAnnealingLR').lower()
    scheduler_params = scheduler_config.get('params', {}) 
    print(f"Initializing Scheduler: {scheduler_name} with params: {scheduler_params}")
    
    if scheduler_name == 'cosineannealinglr':
        if 'T_max' not in scheduler_params:
             scheduler_params['T_max'] = num_epochs  # Default to num_epochs
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    
    elif scheduler_name == 'steplr':
        if 'step_size' not in scheduler_params: 
             scheduler_params['step_size'] = max(1, num_epochs // 3)
             print(f"  (Using default step_size: {scheduler_params['step_size']})")
        if 'gamma' not in scheduler_params:
             scheduler_params['gamma'] = 0.1
             print(f"  (Using default gamma: {scheduler_params['gamma']})")
        scheduler = lr_scheduler.StepLR(optimizer, **scheduler_params)
    
    elif scheduler_name == 'reducelronplateau':
         # Default params for ReduceLROnPlateau if not provided
         if 'mode' not in scheduler_params: scheduler_params['mode'] = 'max' # Maximize EV
         if 'factor' not in scheduler_params: scheduler_params['factor'] = 0.1
         if 'patience' not in scheduler_params: scheduler_params['patience'] = patience // 2 # Scheduler patience < early stopping
         scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    # --- Training Loop ---
    model.to(device)

    train_losses, val_losses = [], []
    train_evs, val_evs = [], []

    best_val_ev = -float('inf')  # initial best EV to avoid early stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0  # counter for early stopping

    print(f"--- Starting Training ---")
    print(f"Epochs: {num_epochs}, Early Stopping Patience: {patience}")
    print(f"-------------------------")

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for images, labels, object_cat in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * images.size(0)
        epoch_train_loss /= len(train_loader.dataset) # Avg loss per sample

        # Evaluation phase
        model.eval()
        epoch_val_loss = 0.0
        preds_val_list, targs_val_list = [], []
        with torch.no_grad():
            # Validation metrics
            for images, labels, object_cat in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.to(device))
                epoch_val_loss += loss.item() * images.size(0)
                preds_val_list.append(outputs.cpu().numpy())
                targs_val_list.append(labels.numpy())

        epoch_val_loss /= len(val_loader.dataset) # Avg loss per sample
        preds_val = np.concatenate(preds_val_list)
        targs_val = np.concatenate(targs_val_list)
        epoch_val_ev = explained_variance_score(targs_val, preds_val)

        # Training Metrics (comment this part for faster training)
        epoch_train_ev = -1.0 
        preds_train_list, targs_train_list = [], []
        with torch.no_grad():
            for images, labels, object_cat in train_loader:
                images = images.to(device)
                outputs = model(images)
                preds_train_list.append(outputs.cpu().numpy())
                targs_train_list.append(labels.numpy())
        preds_train = np.concatenate(preds_train_list)
        targs_train = np.concatenate(targs_train_list)
        epoch_train_ev = explained_variance_score(targs_train, preds_train)

        # Scheduler Step 
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_val_ev) # Use validation EV
        else:
            scheduler.step()

        # Store metrics
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_evs.append(epoch_train_ev) 
        val_evs.append(epoch_val_ev)

        # Print epoch metrics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{num_epochs} | LR={current_lr:.1e} "
              f"Val Loss={epoch_val_loss:.4f} " 
              f"Train EV={epoch_train_ev:.4f} "
              f"Val EV={epoch_val_ev:.4f}" 
              f" | Best Val EV={best_val_ev:.4f}")

        # Early stopping check
        if epoch_val_ev > best_val_ev:
            best_val_ev = epoch_val_ev
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"  (New best model saved with Val EV: {best_val_ev:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    print(f"Training finished. Loading best model with Val EV={best_val_ev:.4f}")
    model.load_state_dict(best_model_wts)

    # Prepare metrics dictionary
    metrics_history = {'train_loss': train_losses, 'val_loss': val_losses,
                       'train_ev': train_evs, 'val_ev': val_evs}

    return model, metrics_history, best_val_ev


# =============================================================================
# UTILS : General Utility Functions (loading data, setting seeds, computing metrics, getting group labels, printing data info)
# =============================================================================

def load_it_data(path_to_data):
    """ Load IT data

    Args:
        path_to_data (str): Path to the data

    Returns:
        np.array (x6): Stimulus train/val/test; objects list train/val/test; spikes train/val
    """
    datafile = h5py.File(os.path.join(path_to_data,'IT_data.h5'), 'r')

    stimulus_train = datafile['stimulus_train'][()]
    spikes_train = datafile['spikes_train'][()]
    objects_train = datafile['object_train'][()]
    
    stimulus_val = datafile['stimulus_val'][()]
    spikes_val = datafile['spikes_val'][()]
    objects_val = datafile['object_val'][()]
    
    stimulus_test = datafile['stimulus_test'][()]
    objects_test = datafile['object_test'][()]

    ### Decode back object type to latin
    objects_train = [obj_tmp.decode("latin-1") for obj_tmp in objects_train]
    objects_val = [obj_tmp.decode("latin-1") for obj_tmp in objects_val]
    objects_test = [obj_tmp.decode("latin-1") for obj_tmp in objects_test]

    return stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val

def download_it_data(path_to_data):
    os.makedirs(path_to_data, exist_ok=True)
    output = os.path.join(path_to_data,"IT_data.h5")
    if not os.path.exists(output):
        url = "https://drive.google.com/file/d/1s6caFNRpyR9m7ZM6XEv_e8mcXT3_PnHS/view?usp=share_link"
        gdown.download(url, output, quiet=False, fuzzy=True)
    else:
        print("File already exists. Skipping download.")

def get_data():
    """
    Downloads, loads, and returns Inferior Temporal (IT) cortex neural response data 
    along with associated visual stimuli and object labels for training, validation, 
    and testing.

    The function performs the following steps:
    - Downloads the IT dataset (if not already present) to the local 'data/' directory.
    - Loads stimulus images, object labels, and neural spike data for train/val/test splits.
    - Prints the unique object categories in the training set.
    - Prints summary information about the training data.

    Returns:
    - stimulus_train (np.ndarray): Training images.
    - stimulus_val (np.ndarray): Validation images.
    - stimulus_test (np.ndarray): Test images.
    - objects_train (list): Object labels for training data.
    - objects_val (list): Object labels for validation data.
    - objects_test (list): Object labels for test data.
    - spikes_train (np.ndarray): Neural responses for training stimuli.
    - spikes_val (np.ndarray): Neural responses for validation stimuli.
    """

    # Download and load IT data
    path_to_data = 'data'
    download_it_data(path_to_data)
    
    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

    # List unique training objects
    unique = list(set(objects_train))
    print("Unique objects in training set:", unique)
    print("Number of unique objects:", len(unique))

    # Print data info
    print_data_info(stimulus_train, spikes_train)

    return stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """Sets the seed for DataLoader workers."""
    base_seed = torch.initial_seed()
    seed = (base_seed + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)

def print_data_info(stimulus_train, spikes_train):
    n_stimulus, n_channels, img_size, _ = stimulus_train.shape
    n_bins , n_neurons = spikes_train.shape
    print('The train dataset contains {} stimuli and {} IT neurons'.format(n_stimulus,n_neurons))
    print('Each stimulus have {} channels (RGB)'.format(n_channels))
    print('The size of the image is {}x{}'.format(img_size,img_size))
    print('The number of bins of neuron spiking rate is {}'.format(n_bins))

def compute_metrics(y_true, y_pred,data_type):
    """
    Computes a suite of regression evaluation metrics comparing predicted and true values.

    Metrics computed:
    - R² score (coefficient of determination)
    - Mean Squared Error (MSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE)
    - Explained Variance (both per output and averaged)
    - Pearson correlation coefficient (per output and average)

    Parameters:
    - y_true (np.ndarray): Ground truth target values, shape (n_samples, n_outputs).
    - y_pred (np.ndarray): Predicted values from the model, same shape as y_true.
    - data_type (str): Label or description of the data being evaluated (e.g., "train", "test").
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    ev = explained_variance_score(y_true, y_pred,multioutput='raw_values')
    ev_avg= explained_variance_score(y_true, y_pred,multioutput='uniform_average')
    corr = np.array([pearsonr(y_true[:, i], y_pred[:, i])[0] for i in range(y_true.shape[1])])
    corr_avg = np.mean(corr)
    print(f"Scores {data_type}: R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}, Explained-Variance (uniform avg)={ev_avg:.4f}, Correlation PearsonR (avg)={corr_avg:.4f}")

    return r2 ,mse, mae, mape, ev, ev_avg, corr, corr_avg

def log_metrics_in_csv(
    model_name: str,
    augmented: bool,
    r2: float,
    mse: float,
    mae: float,
    mape: float,
    ev_avg: float,
    corr_avg: float,
    time_comput=float,
    out_path: str = "out/metrics_log.csv"
):
    """
    Logs model evaluation metrics into a CSV file, appending a new row with timestamped results.

    Parameters:
    - model_name (str): Name or identifier of the model.
    - augmented (bool): Whether data augmentation was used (logged as 0 or 1).
    - r2 (float): R-squared (coefficient of determination) score.
    - mse (float): Mean squared error.
    - mae (float): Mean absolute error.
    - mape (float): Mean absolute percentage error.
    - ev_avg (float): Explained variance averaged across outputs.
    - corr_avg (float): Average Pearson correlation between predicted and true values.
    - time_comput (float): Computation time for model training/evaluation (in seconds).
    - out_path (str): Path to the CSV file where metrics will be logged. Defaults to 'out/metrics_log.csv'.

    Behavior:
    - If the CSV file doesn't exist, it creates it and writes the header.
    - Each call appends a new row with the current timestamp and provided metrics.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Prepare log row
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "datetime": timestamp,
        "model": model_name,
        "augment": int(augmented),
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "mape": mape,
        "ev_avg": ev_avg,
        "corr_avg": corr_avg,
        "time_comput": time_comput
    }

    # Check if we need to write headers
    write_header = not os.path.exists(out_path)

    with open(out_path, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())

        if write_header:
            writer.writeheader()

        writer.writerow(row)

def get_group_labels(object_labels):
    """
    Given a list of object labels, return their high-level group labels
    (e.g., car, fruit, animal, face, etc.)
    """
    fruit_set = {
        'apple', 'apricot', 'strawberry', 'watermelon', 'raspberry', 'pear', 'peach', 'walnut'
    }
    animal_set = {
        'bear', 'dog', 'elephant', 'lioness', 'hedgehog', 'cow', 'turtle', 'gorilla'
    }

    def assign(label):
        if label.startswith('car_'):
            return 'car'
        elif label in fruit_set:
            return 'fruit'
        elif label in animal_set:
            return 'animal'
        elif label.startswith('face'):
            return 'face'
        elif label.startswith('chair'):
            return 'chair'
        elif label.startswith('table'):
            return 'table'
        elif label.startswith('ship'):
            return 'ship'
        elif label.startswith('airplane'):
            return 'airplane'
        else:
            return 'other'

    return np.array([assign(label) for label in object_labels])


# =============================================================================
# UTILS : Plot functions
# =============================================================================

def set_plot_color(model_type):
    """
    Set color for plots based on model type.
    """
    if 'linear' in model_type or 'ridge' in model_type:
        return 'blue'
    elif 'pretrained' in model_type:
        return 'red'
    elif 'random' in model_type:
        return 'orange'
    elif 'data' in model_type:
        return 'violet'
    elif 'ResNet' in model_type or 'VGG' in model_type:
        return 'green'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def plot_corr_ev_distribution(r_values, ev_values, model_name): 
    """Plot the distribution of Pearson correlation coefficients and explained variance scores for all neurons as percentages."""

    # Convert values to percentages
    r_values_percent = r_values * 100
    ev_values_percent = ev_values * 100

    # Set font size and style
    plt.rcParams['font.size'] = 14

    color = set_plot_color(model_name)
    fig, ax = plt.subplots(2, 2, figsize=(7, 3), gridspec_kw={'height_ratios': [1, 3]})

    # --- Correlation Coefficient Boxplot ---
    sns.boxplot(x=r_values_percent, color=color, ax=ax[0, 0], linecolor='black', linewidth=1, width=0.5)
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("")
    ax[0, 0].set_xlim(0, 100)
    ax[0, 0].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for spine in ax[0, 0].spines.values():
        spine.set_visible(False)

    # --- Correlation Coefficient Histogram ---
    sns.histplot(r_values_percent, color=color, ax=ax[1, 0])
    ax[1, 0].set_xlim(0, 100)
    ax[1, 0].set_ylim(0, 50)
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].spines['right'].set_visible(False)

    # --- Explained Variance Boxplot ---
    sns.boxplot(x=ev_values_percent, color=color, ax=ax[0, 1], linecolor='black', linewidth=1, width=0.5)
    ax[0, 1].set_xlabel("")
    ax[0, 1].set_ylabel("")
    ax[0, 1].set_xlim(0, 100)
    ax[0, 1].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for spine in ax[0, 1].spines.values():
        spine.set_visible(False)

    # --- Explained Variance Histogram ---
    sns.histplot(ev_values_percent, color=color, ax=ax[1, 1])
    ax[1, 1].set_xlim(0, 100)
    ax[1, 1].set_ylim(0, 50)
    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs("out/corr_exp_var_histogram", exist_ok=True)
    outpath = f"out/corr_exp_var_histogram/{model_name}_hist.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved correlation and explained variance histograms to: {outpath}")

def plot_population_rdm_analysis(predicted_responses, true_responses, object_labels, model_name, metric='correlation'):
    """
    Computes Representational Dissimilarity Matrices (RDMs) for predicted and true neural population
    responses, applies rank normalization, and plots the resulting RDMs with category-based group
    labels.

    The function performs the following steps:
    - Sorts responses and labels by semantic groupings (e.g., 'car', 'fruit').
    - Computes pairwise distances to form predicted and true RDMs.
    - Applies rank normalization to both RDMs (values scaled to [0, 1] based on percentile ranks).
    - Computes Spearman correlation between the predicted and true RDMs.
    - Generates and saves heatmap plots of the RDMs with annotated group ticks.

    Parameters:
    - predicted_responses (np.ndarray): Model-predicted response matrix of shape (n_objects, features).
    - true_responses (np.ndarray): Ground truth response matrix of the same shape as predicted_responses.
    - object_labels (list or np.ndarray): Labels corresponding to each object/row in the response matrices.
    - model_name (str): Name of the model, used to determine output filenames and optionally condition plotting.
    - metric (str): Distance metric to use for RDM computation (default is 'correlation').

    Returns:
    - spearman_rho (float): Spearman correlation coefficient between predicted and true RDMs.
    """
    assert predicted_responses.shape == true_responses.shape, "Shapes must match"
    n = predicted_responses.shape[0]
    # Set font to Arial and size 16
    plt.rcParams['font.size'] = 14
    # Sort inputs by object labels
    object_labels = np.array(object_labels, dtype=str)
    group_labels = np.array(get_group_labels(object_labels), dtype=str)

    # Sort by group_labels first, then by object_labels within each group
    sorted_indices = np.lexsort((object_labels, group_labels))

    # Apply sorted order
    predicted_responses = predicted_responses[sorted_indices]
    true_responses = true_responses[sorted_indices]
    object_labels = object_labels[sorted_indices]
    group_labels = group_labels[sorted_indices]

    # Compute RDMs
    rdm_pred = pairwise_distances(predicted_responses, metric=metric)
    rdm_true = pairwise_distances(true_responses, metric=metric)

    # Compute Spearman similarity
    triu = np.triu_indices(n, k=1)
    vec_pred = rdm_pred[triu]
    vec_true = rdm_true[triu]
    spearman_rho, _ = spearmanr(vec_pred, vec_true)

    def rank_normalize(rdm):
        """
        Applies rank normalization to the upper triangular part of a Representational 
        Dissimilarity Matrix (RDM), converting dissimilarities to percentiles.

        Parameters:
        - rdm (np.ndarray): A square RDM matrix with dissimilarity values.

        Returns:
        - rdm_ranked (np.ndarray): The RDM with rank-normalized values in the range [0, 1].
        """
        values = rdm[triu]
        ranks = rankdata(values, method='average')
        percentiles = (ranks - 1) / (len(ranks) - 1)
        rdm_ranked = np.zeros_like(rdm)
        rdm_ranked[triu] = percentiles
        rdm_ranked[(triu[1], triu[0])] = percentiles
        return rdm_ranked

    rdm_pred_ranked = rank_normalize(rdm_pred)
    rdm_true_ranked = rank_normalize(rdm_true)

    def plot_rdm(rdm, title, filename):
        """
        Plots a Representational Dissimilarity Matrix (RDM) with group-based tick labels
        and saves it to a specified file.

        Parameters:
        - rdm (np.ndarray): The RDM to plot (2D square matrix with values between 0 and 1).
        - title (str): Title for the plot.
        - filename (str): File path to save the resulting image (PNG format).
        """

        plt.figure(figsize=(6, 5))
        plt.imshow(rdm, cmap='jet', vmin=0, vmax=1)
        #plt.title(f"{title} (Spearman ρ = {spearman_rho:.3f})")

        # Compute tick positions
        ticks = []
        tick_labels = []
        last_label = None
        for i, label in enumerate(group_labels):
            if label != last_label:
                indices = np.where(group_labels == label)[0]
                center = (indices[0] + indices[-1]) // 2
                ticks.append(center)
                tick_labels.append(label)
                last_label = label

        plt.yticks(ticks, tick_labels, fontsize=18)
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")
        plt.xticks([])  # optional: hide cluttered x-axis ticks

        plt.tight_layout()
        os.makedirs("out/rdm", exist_ok=True)
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved RDM plot to {filename}")

    # Plot and save each RDM
    if 'conv1' in model_name:
        plot_rdm(rdm_true_ranked, "True RDM", f"out/rdm/{model_name.split('_')[0]}_true_rdm.png")
    plot_rdm(rdm_pred_ranked, "Predicted RDM", f"out/rdm/{model_name}_predicted_rdm.png")

    return spearman_rho

def plot_neuron_site_response(predicted_responses, true_responses, object_labels, site_index, model_name):
    """
    Plots true vs. predicted spike rates for a specific neuron (site) across stimuli,
    ordered by group and object label.

    Parameters:
    - predicted_responses: (num_stimuli, num_sites)
    - true_responses: (num_stimuli, num_sites)
    - object_labels: list or array of object label strings
    - site_index: integer index of the neuron/site to plot
    - model_name: string for saving the output figure
    """
    assert predicted_responses.shape == true_responses.shape
    assert 0 <= site_index < predicted_responses.shape[1]

    # Get group labels
    object_labels = np.array(object_labels)
    group_labels = get_group_labels(object_labels)

    # Sort by group then object label
    sorted_indices = np.lexsort((object_labels, group_labels))
    object_labels = object_labels[sorted_indices]
    group_labels = group_labels[sorted_indices]
    y_true = true_responses[sorted_indices, site_index]
    y_pred = predicted_responses[sorted_indices, site_index]

    # Plotting
    color = set_plot_color(model_name)
    plt.figure(figsize=(6, 4))
    plt.plot(y_true, label="True", color='black', linewidth=1.5, alpha=0.7)
    plt.plot(y_pred, label="Predicted", color=color, linewidth=2, alpha=0.8)
    #plt.ylabel("Spike Rate")
    #plt.title(f"Neural Site {site_index} Response - {model_name}")

    # Vertical dashed lines for group changes
    last_group = group_labels[0]
    for i in range(1, len(group_labels)):
        if group_labels[i] != last_group:
            plt.axvline(x=i, color='gray', linestyle='--', linewidth=0.8)
            last_group = group_labels[i]

    # Show group label ticks centered per group (labels only, no tick marks)
    ticks = []
    tick_labels = []
    last_label = group_labels[0]
    start = 0
    for i in range(1, len(group_labels)):
        if group_labels[i] != last_label:
            center = (start + i - 1) // 2
            ticks.append(center)
            tick_labels.append(last_label)
            start = i
            last_label = group_labels[i]
    # Handle final group
    center = (start + len(group_labels) - 1) // 2
    ticks.append(center)
    tick_labels.append(last_label)

    if 'ridge' in model_name:
        plt.xticks(ticks, tick_labels, rotation=45,ha='center', fontsize=14)
    else:
        plt.xticks([])
    plt.tick_params(axis='x', length=0)  # Hide tick marks but keep labels
    plt.xlabel("")  # Remove x-axis label

    # Remove top and right borders only
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save
    plt.tight_layout()
    os.makedirs("out/neuron_site", exist_ok=True)
    outpath = f"out/neuron_site/{model_name}_site{site_index}.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved neuron response plot to: {outpath}")

def visualize_img(stimulus,objects,stim_idx):
    """Visualize image given the stimulus and corresponding index and the object name.

    Args:
        stimulus (array of float): Stimulus containing all the images
        objects (list of str): Object list containing all the names
        stim_idx (int): Index of the stimulus to plot
    """    
    normalize_mean=[0.485, 0.456, 0.406]
    normalize_std=[0.229, 0.224, 0.225]

    img_tmp = np.transpose(stimulus[stim_idx],[1,2,0])

    ### Go back from normalization
    img_tmp = (img_tmp*normalize_std + normalize_mean) * 255

    plt.figure()
    plt.imshow(img_tmp.astype(np.uint8),cmap='gray')
    plt.title(str(objects[stim_idx]))
    plt.show()
    return

def plot_layer_comparison(ev_scores, layers_of_interest, model_type):
    """
    Generates a bar plot comparing explained variance scores across different layers of a model.
    
    Args:
        ev_scores (list or np.ndarray): List of explained variance scores (as fractions or percentages).
        layers_of_interest (list): List of layer names or identifiers.
        model_type (str): Type or name of the model (e.g., 'ResNet', 'VGG').

    Saves the plot to: out/layer_comparison/{model_type}_layer_comparison.png
    """
    # Convert EV scores to percentage if they appear to be in [0, 1]
    if max(ev_scores) <= 1.0:
        ev_scores = [score * 100 for score in ev_scores]

    plt.figure(figsize=(10, 5))
    plt.bar(layers_of_interest, ev_scores, color=set_plot_color(model_type))
    plt.ylim(0, 40)  # y-axis limit set from 0 to 40
    plt.xticks(rotation=45, ha='right')
    
    # Style tweaks
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs("out/layer_comparison", exist_ok=True)
    outpath = f"out/layer_comparison/{model_type}_layer_comparison.png"
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"Saved layer comparison plot to: {outpath}")



# =============================================================================
# RUN : Model Function Definitions
# =============================================================================

# =======  PART 1a =======  
def run_linear(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Linear Regression model...")

    # Vectorize input data
    X_train = stimulus_train.reshape(stimulus_train.shape[0], -1)
    X_val = stimulus_val.reshape(stimulus_val.shape[0], -1)

    # Train model on full training set
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, spikes_train)
    computation_time = time.time() - start_time
    print(f"done. Computation time: {computation_time:.4f}")

    # Evaluate on validation set (optional)
    spikes_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(spikes_val, spikes_val_pred)
    print(f"Validation MSE with best Ridge model: {val_mse:.4f}")

    # Compute metrics for training and validation set
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name="linear_val",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val, 'linear')

    # plot dissimilarity matrix on validation set
    plot_population_rdm_analysis(spikes_val_pred,spikes_val, objects_val, 'linear', metric='correlation')

    # plot neuron site brain activity prediction
    for site in [39,107,152]:
        plot_neuron_site_response(spikes_val_pred,spikes_val, objects_val, site, 'linear')

def run_linear_pca(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Linear Regression with PCA...")

    # Compute PCA
    X_train, X_val = get_pca(stimulus_train.reshape(stimulus_train.shape[0], -1), stimulus_val.reshape(stimulus_val.shape[0], -1), n_components=1000, model_type='linear_model')

    # Train model on full training set
    start_time = time.time()
    model = LinearRegression()
    model.fit(X_train, spikes_train)
    computation_time = time.time() - start_time
    print(f"done. Computation time: {computation_time:.4f}")

    # Evaluate on validation set (optional)
    spikes_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(spikes_val, spikes_val_pred)
    print(f"Validation MSE with best Ridge model: {val_mse:.4f}")

    # Compute metrics for training and validation set
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name="linear_pca_val",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val, 'linear_pca')

    # plot dissimilarity matrix on validation set
    plot_population_rdm_analysis(spikes_val_pred,spikes_val, objects_val, 'linear_pca', metric='correlation')

    # plot neuron site brain activity prediction
    for site in [39,107,152]:
        plot_neuron_site_response(spikes_val_pred,spikes_val, objects_val, site, 'linear_pca')

def run_ridge_cv5(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Ridge Regression with 5-fold CV...")

    X_train = stimulus_train.reshape(stimulus_train.shape[0], -1)
    X_val = stimulus_val.reshape(stimulus_val.shape[0], -1)

    # Ridge regression setup
    alphas = np.logspace(5, 6, 10)
    ridge = Ridge()

    # Stratified K-Fold using class labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search with stratified CV
    grid = GridSearchCV(
        estimator=ridge,
        param_grid={'alpha': alphas},
        scoring='neg_mean_squared_error',
        cv=skf.split(X_train, objects_train),
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    start_time = time.time()

    grid.fit(X_train, spikes_train)

    computation_time = time.time() - start_time
    print(f"Grid Search Computation time: {computation_time:.4f}")
    
    # Best model and parameter
    print(f"\nBest alpha: {grid.best_params_['alpha']}")
    print(f"Best CV MSE: {-grid.best_score_:.4f}")

    # Train best model on full training set
    start_time = time.time()
    model = Ridge(alpha=grid.best_params_['alpha'])
    model.fit(X_train, spikes_train)
    computation_time = time.time() - start_time
    print(f"done. Computation time: {computation_time:.4f}")

    # Evaluate on validation set (optional)
    spikes_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(spikes_val, spikes_val_pred)
    print(f"Validation MSE with best Ridge model: {val_mse:.4f}")

    # Compute metrics for training and validation set
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name="ridge_cv5_val",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,'ridge_cv5')

    # plot dissimilarity matrix on validation set
    plot_population_rdm_analysis(spikes_val_pred,spikes_val, objects_val, 'ridge_cv5', metric='correlation')

    # plot neuron site brain activity prediction
    for site in [39,107,152]:
        plot_neuron_site_response(spikes_val_pred,spikes_val, objects_val, site, 'ridge_cv5')

def run_ridge_pca_cv5(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Ridge Regression with PCA and 5-fold CV...")

    # Normalize data
    """ scaler = StandardScaler()
    stimulus_train = scaler.fit_transform(stimulus_train.reshape(stimulus_train.shape[0], -1))
    stimulus_val = scaler.transform(stimulus_val.reshape(stimulus_val.shape[0], -1))
    stimulus_test = scaler.transform(stimulus_test.reshape(stimulus_test.shape[0], -1)) """
    # Compute PCA
    X_train, X_val = get_pca(stimulus_train.reshape(stimulus_train.shape[0], -1), 
                             stimulus_val.reshape(stimulus_val.shape[0], -1), 
                             n_components=1000, model_type='ridge_model')

    # Ridge regression setup
    alphas = np.logspace(5,6,10)
    ridge = Ridge()

    # Stratified K-Fold using class labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search with stratified CV
    grid = GridSearchCV(
        estimator=ridge,
        param_grid={'alpha': alphas},
        scoring='neg_mean_squared_error',
        cv=skf.split(X_train, objects_train),
        n_jobs=-1,
        verbose=1
    )

    # Fit grid search
    start_time = time.time()

    grid.fit(X_train, spikes_train)

    computation_time = time.time() - start_time
    print(f"Grid Search Computation time: {computation_time:.4f}")
    
    # Best model and parameter
    print(f"\nBest alpha: {grid.best_params_['alpha']}")
    print(f"Best CV MSE: {-grid.best_score_:.4f}")

    # Train best model on full training set
    start_time = time.time()
    model = Ridge(alpha=grid.best_params_['alpha'])
    model.fit(X_train, spikes_train)
    computation_time = time.time() - start_time
    print(f"done. Computation time: {computation_time:.4f}")

    # Evaluate on validation set (optional)
    spikes_val_pred = model.predict(X_val)
    spikes_train_pred = model.predict(X_train)
    val_mse = mean_squared_error(spikes_val, spikes_val_pred)
    print(f"Validation MSE with best Ridge model: {val_mse:.4f}")
    print(spikes_val_pred.shape)

    # Compute metrics for training and validation set
    (r2_train, mse_train, mae_train, mape_train, ev_train, ev_avg_train, corr_train, corr_avg_train) = compute_metrics(spikes_train, spikes_train_pred,'train')
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name='ridge_pca_cv5_val', augmented=augmented, r2=r2_val, mse=mse_val, mae=mae_val, mape=mape_val, ev_avg=ev_avg_val, corr_avg=corr_avg_val, time_comput=computation_time)
    log_metrics_in_csv(model_name='ridge_pca_cv5_train', augmented=augmented, r2=r2_train, mse=mse_train, mae=mae_train, mape=mape_train, ev_avg=ev_avg_train, corr_avg=corr_avg_train, time_comput=computation_time)
    
    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,'ridge_pca_cv5')

    # plot dissimilarity matrix on validation set
    plot_population_rdm_analysis(spikes_val_pred,spikes_val, objects_val, 'ridge_pca_cv5', metric='correlation')

    # plot neuron site brain activity prediction
    for site in [39,107,152]:
        plot_neuron_site_response(spikes_val_pred,spikes_val, objects_val, site, 'ridge_pca_cv5')

# =======  PART 1b ========
def run_task_driven(model,device,stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented,model_type,verbose=True):
    print(f"=== Running Task-driven model ({model_type} weights) using {'augmented' if augmented else 'original'} dataset ===")
    
    # === Parameters ====
    DATA_PATH =f'data/task_driven_models'
    BATCH_SIZE = 100
    N_PCA = 1000
    # ===================

    layers_of_interest = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

     # Create output directories for saving activations and PCA models
    save_dir_pc =  os.path.join(DATA_PATH,f"{N_PCA}pcs",f"{model_type}_pca")
    os.makedirs(save_dir_pc, exist_ok=True)

    # Set up feature extractor
    feature_extractor = create_feature_extractor(model, return_nodes={layer: layer for layer in layers_of_interest}).to(device)
    feature_extractor.eval()

    # === Collect activations ===
    def collect_activations(loader):
        activations = []
        for batch_x, _ in tqdm(loader, desc=f"Extracting {layer_name}"):
            with torch.no_grad():
                features = feature_extractor(batch_x)[layer_name]
                activations.append(features.view(features.size(0), -1).cpu().numpy())
        return np.concatenate(activations, axis=0)
    # === Prepare data loaders ===
    def get_dataloader(stimulus, spikes):
        inputs = torch.tensor(stimulus).to(device)
        outputs = torch.tensor(spikes).to(device)
        dataset = TensorDataset(inputs, outputs)
        return DataLoader(dataset, batch_size=BATCH_SIZE)
    # =======================================================================================
    
    train_loader = get_dataloader(stimulus_train, spikes_train)
    val_loader = get_dataloader(stimulus_val, spikes_val)

    compute_pcs = len(os.listdir(save_dir_pc)) < len(layers_of_interest) * 3
    if compute_pcs:
        # === Extract activations and apply PCA ===
        for layer_name in layers_of_interest:
            print(f"\nProcessing layer: {layer_name} ===")

            X_train_acts = collect_activations(train_loader)
            X_val_acts = collect_activations(val_loader)

            # === PCA ===
            print("Fitting PCA on train activations...")
            pca = PCA(n_components=N_PCA)
            X_train_pcs = pca.fit_transform(X_train_acts)
            X_val_pcs = pca.transform(X_val_acts)

            # === Save PCA model and PCs ===
            pca_path = os.path.join(save_dir_pc, f'{layer_name}_pca_model.pkl')
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)

            np.save(os.path.join(save_dir_pc, f'{layer_name}_train_pcs.npy'), X_train_pcs)
            np.save(os.path.join(save_dir_pc, f'{layer_name}_val_pcs.npy'), X_val_pcs)

            print(f"Saved PCA model and PCs for {layer_name}")

    
    # ======== Evaluate the taks-driven model using the PCA-reduced representations ======== 
    print("Apply PCs to activations and fit a linear Ridge regression model to predict brain activity")


    ev_scores = []
    for layer_name in layers_of_interest:

        # Load precomputed PCs
        X_train = np.load(os.path.join(save_dir_pc, f'{layer_name}_train_pcs.npy'))
        X_val = np.load(os.path.join(save_dir_pc, f'{layer_name}_val_pcs.npy'))


        # ======= Ridge regression setup ======= 
        print(f"\n{layer_name}\n===============")
        #print(f"{layer_name} Linear Regression using Ridge (grid search on alpha)")
        alphas = np.logspace(-3,6, 100)
        ridge = Ridge(max_iter=1000,fit_intercept=True,random_state=42)

        # Stratified K-Fold using class labels
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Grid search with stratified CV
        grid = GridSearchCV(
            estimator=ridge,
            param_grid={'alpha': alphas},
            scoring='neg_mean_squared_error',
            cv=skf.split(X_train, objects_train),
            n_jobs=-1,
            verbose=0
        )

        # Fit grid search
        start_time = time.time()

        grid.fit(X_train, spikes_train)

        computation_time = time.time() - start_time
        #print(f"Grid Search Computation time: {computation_time:.4f}")
        
        # Best model and parameter
        print(f"Best alpha: {grid.best_params_['alpha']}")
        #print(f"Best CV MSE: {-grid.best_score_:.4f}")

        # Train best model on full training set
        start_time = time.time()
        ridge_model = Ridge(alpha=grid.best_params_['alpha'],max_iter=1000,fit_intercept=True,random_state=42)
        ridge_model.fit(X_train, spikes_train)
        computation_time = time.time() - start_time
        #print(f"done. Computation time: {computation_time:.4f}")

        # Evaluate on validation set (optional)
        spikes_val_pred = ridge_model.predict(X_val)
        spikes_train_pred = ridge_model.predict(X_train)
        val_mse = mean_squared_error(spikes_val, spikes_val_pred)
        #print(f"Validation MSE with best Ridge model: {val_mse:.4f}")

        # Compute metrics for training and validation set
        (r2_train, mse_train, mae_train, mape_train, ev_train, ev_avg_train, corr_train, corr_avg_train) = compute_metrics(spikes_train, spikes_train_pred,'train')
        (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
        log_metrics_in_csv(model_name=f'task_driven_val_{model_type}_{layer_name}', augmented=augmented, r2=r2_val, mse=mse_val, mae=mae_val, mape=mape_val, ev_avg=ev_avg_val, corr_avg=corr_avg_val, time_comput=computation_time)
        log_metrics_in_csv(model_name=f'task_driven_train_{model_type}_{layer_name}', augmented=augmented, r2=r2_train, mse=mse_train, mae=mae_train, mape=mape_train, ev_avg=ev_avg_train, corr_avg=corr_avg_train, time_comput=computation_time)
        ev_scores.append(ev_avg_val)

       # plot correlation and explained-variance distribution
        plot_corr_ev_distribution(corr_val, ev_val, f'task_driven_{model_type}_{layer_name}')

        # plot dissimilarity matrix on validation set
        plot_population_rdm_analysis(spikes_val_pred,spikes_val, objects_val, f'task_driven_{model_type}_{layer_name}', metric='correlation')

        # plot neuron site brain activity prediction
        #for site in [23,39,42,56,71,95,107,132,139,152]:
        for site in [39,107,152]:
            plot_neuron_site_response(spikes_val_pred,spikes_val, objects_val, site, f'task_driven_{model_type}_{layer_name}')

    plot_layer_comparison(ev_scores, layers_of_interest, model_type)

def run_task_driven_pretrained(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    run_task_driven(model,device,stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented,model_type='pretrained')

def run_task_driven_random(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=None).to(device)
    run_task_driven(model, device, stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val, augmented,model_type='random')


# ======= PART 2  =======
def run_data_driven(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Data-driven model...")

    data_path = 'data'
    stim_train, stim_val, *_ , spikes_train, spikes_val = load_it_data(data_path)
    _, _, H, _ = stim_train.shape
    _, n_neurons = spikes_train.shape

    train_ds = ITDataset(stim_train, spikes_train, objects_train)
    val_ds   = ITDataset(stim_val,   spikes_val, objects_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShallowCNN(n_neurons).to(device)

    start_time = time.time()
    print("Training model...")
    model, metrics = train_model(model, train_loader, val_loader, device,
                                 num_epochs=40, learning_rate=1e-3)
    computation_time = time.time() - start_time
    spikes_val_pred, targs_val, objects_val = evaluate_model(model, val_loader, device)
    spikes_train_pred, targs_train, objects_train = evaluate_model(model, train_loader, device)
    #save_training_plots(metrics)

    # Compute metrics for training and validation set
    (r2_train, mse_train, mae_train, mape_train, ev_train, ev_avg_train, corr_train, corr_avg_train) = compute_metrics(targs_train, spikes_train_pred,'train')
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(targs_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name='data_driven_shallow_cnn', augmented=augmented, r2=r2_val, mse=mse_val, mae=mae_val, mape=mape_val, ev_avg=ev_avg_val, corr_avg=corr_avg_val, time_comput=computation_time)
    log_metrics_in_csv(model_name='data_driven_shallow_cnn', augmented=augmented, r2=r2_train, mse=mse_train, mae=mae_train, mape=mape_train, ev_avg=ev_avg_train, corr_avg=corr_avg_train, time_comput=computation_time)
    
    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,'data_driven_shallow_cnn')

    # plot dissimilarity matrix on validation set
    plot_population_rdm_analysis(spikes_val_pred, targs_val, objects_val, 'data_driven_shallow_cnn', metric='correlation')

    # plot neuron site brain activity prediction
    for site in [39,107,152]:
        plot_neuron_site_response(spikes_val_pred,targs_val, objects_val, site, 'data_driven_shallow_cnn')

# ======= PART 3  =======
def run_experiment(config, n_neurons, train_loader, val_loader, device):
    """Runs a single training experiment with the given configuration."""
    exp_name = config.get('name', f"exp_{int(time.time())}")
    print(f"\n===== Starting Experiment: {exp_name} =====")
    start_time = time.time()

    current_seed = config.get('seed', SEED)
    set_seed(current_seed)
    print(f"Using Seed: {current_seed}")

    # --- Initialize Model based on architecture ---
    model_config = config.get('model_config', {})
    architecture = model_config.get('architecture', 'resnet').lower()  # Default to ResNet if not specified
    pretrained = model_config.get('pretrained', True)                  # Default to pretrained weights
    freeze_features = model_config.get('freeze_features', False)       # Default to full training
    unfreeze_blocks = model_config.get('unfreeze_blocks', 0)           # Default to 0 blocks
    classifier_config = model_config.get('classifier', None)           # Default to classifier implemented in each model class
    
    if architecture == 'resnet':
        resnet_version = model_config.get('resnet_version', 'resnet34')  # Default to ResNet34
        model = ResNetAdapter(
            n_neurons,
            resnet_version=resnet_version,
            pretrained=pretrained,
            classifier_config=classifier_config,
            freeze_features=freeze_features,
            unfreeze_blocks=unfreeze_blocks
        )
        print(f"Model: {resnet_version.upper()} Adapter")
    elif architecture == 'vgg_bn':
        model = VGG_BN(
            n_neurons,
            pretrained=pretrained,
            classifier_config=classifier_config,
            freeze_features=freeze_features,
            unfreeze_blocks=unfreeze_blocks
        )
        print(f"Model: VGG_BN")
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    
    print(f"Model Config: {model_config}")

    # --- Training parameters ---
    train_params = config.get('train_params', {})
    num_epochs = train_params.get('num_epochs', 30)
    patience = train_params.get('patience', 5)

    optimizer_config = config.get('optimizer', {'name': 'AdamW', 'params': {'lr': 1e-4, 'weight_decay': 1e-5}})
    scheduler_config = config.get('scheduler', {'name': 'CosineAnnealingLR', 'params': {}})

    # --- Train the model ---
    start_time = time.time()
    model, metrics, best_val_ev = train_model_best(
        model,
        train_loader,
        val_loader,
        device,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        num_epochs=num_epochs,
        patience=patience
    )
    computation_time = time.time() - start_time
    # --- Evaluate the model ---
    print("\nFinal evaluation on validation set using best model:")
    spikes_train_pred, targs_train, objects_train = evaluate_model(model, train_loader, device)
    spikes_val_pred, targs_val, objects_val = evaluate_model(model, val_loader, device)

    end_time = time.time()
    duration = end_time - start_time
    
    # Compute metrics for training and validation set
    (r2_train, mse_train, mae_train, mape_train, ev_train, ev_avg_train, corr_train, corr_avg_train) = compute_metrics(targs_train, spikes_train_pred,'train')
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(targs_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name= exp_name + '_val', augmented=False, r2=r2_val, mse=mse_val, mae=mae_val, mape=mape_val, ev_avg=ev_avg_val, corr_avg=corr_avg_val, time_comput=computation_time)
    log_metrics_in_csv(model_name=exp_name + '_train', augmented=False, r2=r2_train, mse=mse_train, mae=mae_train, mape=mape_train, ev_avg=ev_avg_train, corr_avg=corr_avg_train, time_comput=computation_time)
    
    print(f"===== Experiment Finished: {exp_name} | Best Val EV: {best_val_ev:.4f} | Final Val EV: {ev_avg_val:.4f} | Duration: {duration:.2f}s =====")

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,exp_name)

    # plot dissimilarity matrix on validation set
    plot_population_rdm_analysis(spikes_val_pred, targs_val, objects_val, exp_name, metric='correlation')

    # plot neuron site brain activity prediction
    for site in [39,107,152]:
        plot_neuron_site_response(spikes_val_pred,targs_val, objects_val, site, exp_name)

    # Return metrics 
    return {
        'config_name': exp_name, 
        'architecture': architecture,
        'best_val_ev': best_val_ev, 
        'final_val_ev': ev_val, 
        'final_val_r2': r2_val, 
        'duration_s': duration, 
        'seed': current_seed, 
        'config': config
    }

def run_vgg_bn(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running VGG model (2nd best model)...")
    train_ds = ITDataset(stimulus_train, spikes_train, objects_train, transform=None) 
    val_ds = ITDataset(stimulus_val, spikes_val, objects_val, transform=None)      

    num_workers = 1 
    batch_size = 128 
    g_main = torch.Generator()
    g_main.manual_seed(SEED)
    train_loader = data.DataLoader(train_ds, 
                                   batch_size=batch_size, 
                                   shuffle=True,
                                   num_workers=num_workers, 
                                   worker_init_fn=worker_init_fn, 
                                   generator=g_main)
    val_loader = data.DataLoader(val_ds, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 num_workers=num_workers, 
                                 worker_init_fn=worker_init_fn, 
                                 generator=g_main)

    config = {
             'name': 'VGG_Wider_7000x2_D0.47_StepLR_15_0.5', 
             'model_config': {
                 'architecture': 'vgg_bn',           
                 'pretrained': True,
                 'freeze_features': False,
                 'classifier': {'hidden_sizes': [7000, 7000], 'activation': 'relu', 'dropout': 0.47}
             },
             'optimizer': {'name': 'AdamW', 'params': {'lr': 1e-4, 'weight_decay': 1e-5}},
             'scheduler': {'name': 'StepLR', 'params': {'step_size': 15, 'gamma': 0.5}}, 
             'train_params': {'num_epochs': 60, 'patience': 7}
        }

    _, n_neurons = spikes_train.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _ = run_experiment(config, n_neurons, train_loader, val_loader, device)

def run_resnet_adapter(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running ResNet34 Adapter model (best model)...")
    train_ds = ITDataset(stimulus_train, spikes_train, objects_train, transform=None) 
    val_ds = ITDataset(stimulus_val, spikes_val, objects_val, transform=None)      

    num_workers = 1 # initially 2 but warning
    batch_size = 128 
    g_main = torch.Generator()
    g_main.manual_seed(SEED)
    train_loader = data.DataLoader(train_ds, 
                                   batch_size=batch_size, 
                                   shuffle=True,
                                   num_workers=num_workers, 
                                   worker_init_fn=worker_init_fn, 
                                   generator=g_main)
    val_loader = data.DataLoader(val_ds, 
                                 batch_size=batch_size, 
                                 shuffle=False,
                                 num_workers=num_workers, 
                                 worker_init_fn=worker_init_fn, 
                                 generator=g_main)

    config = {
            'name': 'ResNet34_7000x2_classifier_0.52',
            'model_config': {
                'architecture': 'resnet',            
                'resnet_version': 'resnet34',        
                'pretrained': True,
                'freeze_features': False, 
                'classifier': {'hidden_sizes': [7000, 7000], 'activation': 'relu', 'dropout': 0.52}
            },
            'optimizer': {'name': 'Adam', 'params': {'lr': 1e-4, 'weight_decay': 1e-5}},
            'scheduler': {'name': 'StepLR', 'params': {'step_size': 15, 'gamma': 0.5}}, 
            'train_params': {'num_epochs': 60, 'patience': 10}
        }

    _, n_neurons = spikes_train.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _ = run_experiment(config, n_neurons, train_loader, val_loader, device)

def run_best(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Best model (as selected by validation performance)...")
    run_resnet_adapter(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)

def run_all(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running All models... ")

    augmented = False
    run_linear(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_linear_pca(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_ridge_cv5(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_ridge_pca_cv5(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_task_driven_pretrained(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_task_driven_random(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_data_driven(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_vgg_bn(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)
    run_resnet_adapter(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented)


# =============================================================================
# PARSER DEFINITION
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="NX-414 - Predicting Neural Activity")

    parser.add_argument(
        "--model",
        choices=[
            "linear",
            "linear_pca",
            "ridge_cv5",
            "ridge_pca_cv5",
            "task_driven_pretrained",
            "task_driven_random",
            "data_driven",
            "resnet_adapter",
            "vgg_bn",
            "best",
            "all"
        ],
        default="best",
        help=(
            "Choose the model (default: best):\n"
            "  linear               - Basic Linear Regression\n"
            "  linear_pca           - Linear Regression with PCA\n"
            "  ridge_cv5            - Ridge Regression with 5-fold Cross-Validation\n"
            "  ridge_pca_cv5        - Ridge Regression with PCA and CV\n"
            "  enet_pca_cv5         - Elastic Net Regression with PCA and CV\n"
            "  task_driven_pretrained  - Task-driven model using trained weights\n"
            "  task_driven_random   - Task-driven model using random weights\n"
            "  data_driven          - Data-driven model\n"
            "  resnet_adapter       - ResNet Adapter model\n"
            "  vgg_bn              - VGG model\n"
            "  best                 - Best-performing model => resnet_adapter\n"
            "  all                  - Run all models sequentially and output metrics as CSV"
        )
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use augmented dataset instead of original"
    )

    return parser.parse_args()


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    args = parse_args()

    # Fix seed to 42
    # Set the seed globally 
    SEED = 42
    set_seed(SEED)
    
    # Load data and compute PCA
    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = get_data()

    # run selected model
    model_map = {
        "linear": run_linear,
        "linear_pca": run_linear_pca,
        "ridge_cv5": run_ridge_cv5,
        "ridge_pca_cv5": run_ridge_pca_cv5,
        "task_driven_pretrained": run_task_driven_pretrained,
        "task_driven_random": run_task_driven_random,
        "data_driven": run_data_driven,
        "resnet_adapter": run_resnet_adapter,
        "vgg_bn": run_vgg_bn,
        "best": run_best,
        "all": run_all,
    }

    model_function = model_map.get(args.model)
    if model_function:
        model_function(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,args.augment)
    else:
        print("Invalid model selected.")
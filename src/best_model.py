"""
This script allows to train and evaluate different neural network architectures to predict neural activity in the IT cortex from images.
It is possible to either train from scratch or load pretrained weights optimized on ImageNet classification tasks.
If pretrained weights are used, the model can be frozen or fine-tuned. 
In addition, a modular classifier head is defined to allow the user to maximize the explained variance of the model on the validation set.

Methodology: 
    Different pytorch pretrained models were tested, including AlexNet, VGG, resnet (https://pytorch.org/vision/main/models.html). 
    It was observed that the VGG11_BN and ResNet models performed well in terms of explained variance. 
    The VGG model was modified by varying the number of convolutional layers. Tests with 3, 4, 5, 6 and 7 layers were performed.
    The best performing model was the one with 5 convolutional layers, which is the one used in this script.
"""

import os
import random
import time
import copy
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import vgg11_bn, VGG11_BN_Weights

from utils import load_it_data

# =============================================================================
# REPRODUCIBILITY UTILITIES
# =============================================================================

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

# Set the seed globally 
SEED = 42
set_seed(SEED)

# =============================================================================
# DATASET CLASS
# =============================================================================

class ITDataset(data.Dataset):
    """Dataset for images and neural activity in IT cortex."""
    def __init__(self, stimuli, neural_responses, transform=None):
        self.stimuli = stimuli
        self.neural_responses = neural_responses
        self.transform = transform

    def __len__(self):
        return self.stimuli.shape[0]

    def __getitem__(self, idx):
        img_np = self.stimuli[idx]
        label = self.neural_responses[idx]

        if self.transform:
            # Convert (C, H, W) numpy array to PIL Image (H, W, C)
            img = np.transpose(img_np, (1, 2, 0))
            if img.dtype != np.uint8:
                if img.max() <= 1.0 and img.min() >= 0:
                    img = (img * 255).astype(np.uint8)
                else: 
                    img = img.astype(np.uint8)
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        else:
            img = torch.tensor(img_np, dtype=torch.float32)

        return img, torch.tensor(label, dtype=torch.float32)

# =============================================================================
# MODEL ARCHITECTURES
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

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(model, train_loader, val_loader, device, optimizer_config, scheduler_config, num_epochs=50, patience=10):
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
        for images, labels in train_loader:
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
            for images, labels in val_loader:
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
            for images, labels in train_loader:
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
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, loader, device):
    """Evaluates the final model and returns predictions and true values."""
    model.eval()
    preds, targs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds.append(model(images).cpu().numpy())
            targs.append(labels.numpy())
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    final_r2 = r2_score(targs, preds)
    final_ev = explained_variance_score(targs, preds)
    print(f"Final R²: {final_r2:.4f}")
    print(f"Final Explained Variance: {final_ev:.4f}")
    return preds, targs, final_r2, final_ev

# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def save_training_plots(metrics, folder="plots_experiment"):
    """Saves the training curves."""
    os.makedirs(folder, exist_ok=True)
    epochs = np.arange(1, len(metrics['val_loss']) + 1) # Use val_loss length

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder, 'loss_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    # Only plot train_ev if it was calculated (not -1.0)
    if any(ev != -1.0 for ev in metrics['train_ev']):
        plt.plot(epochs, metrics['train_ev'], 'b-', label='Train EV')
    plt.plot(epochs, metrics['val_ev'], 'r-', label='Val EV')
    plt.title('Explained Variance Curves')
    plt.xlabel('Epochs'); plt.ylabel('Explained Variance'); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(folder, 'ev_curve.png'))
    plt.close()
    print(f"Training plots saved in folder '{folder}'")


# =============================================================================
# EXPERIMENT RUNNER FUNCTION
# =============================================================================

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
    model, metrics, best_val_ev = train_model(
        model,
        train_loader,
        val_loader,
        device,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        num_epochs=num_epochs,
        patience=patience
    )

    # --- Evaluate the model ---
    print("\nFinal evaluation on validation set using best model:")
    val_preds, val_targs, final_r2, final_ev = evaluate_model(model, val_loader, device)

    end_time = time.time()
    duration = end_time - start_time
    print(f"===== Experiment Finished: {exp_name} | Best Val EV: {best_val_ev:.4f} | Final Val EV: {final_ev:.4f} | Duration: {duration:.2f}s =====")

    # --- Save results ---
    output_folder = f"results_{exp_name}_seed{current_seed}"
    os.makedirs(output_folder, exist_ok=True)
    save_training_plots(metrics, folder=output_folder)
    model_filename = os.path.join(output_folder, f"{exp_name}_model_seed{current_seed}.pt")
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'")
    predictions_csv_filename = os.path.join(output_folder, f"predictions_{exp_name}_seed{current_seed}.csv")
    try:
        num_neurons_pred = val_preds.shape[1]
        neuron_cols = [f'neuron_{i}' for i in range(num_neurons_pred)]
        preds_df = pd.DataFrame(val_preds, columns=neuron_cols)
        preds_df.to_csv(predictions_csv_filename, index=False)
        print(f"Validation predictions saved to '{predictions_csv_filename}'")
    except Exception as e:
        print(f"Error saving predictions to CSV {predictions_csv_filename}: {e}")

    # Return metrics 
    return {
        'config_name': exp_name, 
        'architecture': architecture,
        'best_val_ev': best_val_ev, 
        'final_val_ev': final_ev, 
        'final_val_r2': final_r2, 
        'duration_s': duration, 
        'seed': current_seed, 
        'config': config
    }


# =============================================================================
# MAIN FUNCTION - Unified Experiment Runner
# =============================================================================
def main():
    """Main function to run multiple training experiments with different architectures."""
    set_seed(SEED)
    print(f"Global reproducibility seed set to: {SEED}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = 'data'
    try:
        stim_train, stim_val, *_ , spikes_train, spikes_val = load_it_data(data_path)
        _, n_neurons = spikes_train.shape
        print(f"Data loaded: {stim_train.shape[0]} train, {stim_val.shape[0]} val images. Predicting for {n_neurons} neurons.")
    except Exception as e:
        print(f"Error loading data: {e}. Exiting.")
        return

    # --- DataLoaders ---
    # No transformations applied to the images as they are already pre-processed
    # Additionally, no improvement were noticed with various data augmentations
    train_ds = ITDataset(stim_train, spikes_train, transform=None) 
    val_ds = ITDataset(stim_val, spikes_val, transform=None)      

    num_workers = 2
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

    # --- Define Experiment Configurations ---
    experiment_configs = [
        # ResNet Configuration Example
        {
            'name': 'ResNet50_7000x2_classifier_0.52',
            'model_config': {
                'architecture': 'resnet',            
                'resnet_version': 'resnet50',        
                'pretrained': True,
                'freeze_features': False, 
                'classifier': {'hidden_sizes': [7000, 7000], 'activation': 'relu', 'dropout': 0.52}
            },
            'optimizer': {'name': 'Adam', 'params': {'lr': 1e-4, 'weight_decay': 1e-5}},
            'scheduler': {'name': 'StepLR', 'params': {'step_size': 15, 'gamma': 0.5}}, 
            'train_params': {'num_epochs': 60, 'patience': 10}
        },

        # VGG Configuration Example
        # {
        #     'name': 'VGG_Wider_7000x2_D0.47_StepLR_15_0.5', 
        #     'model_config': {
        #         'architecture': 'vgg_bn',           
        #         'pretrained': True,
        #         'freeze_features': False,
        #         'classifier': {'hidden_sizes': [7000, 7000], 'activation': 'relu', 'dropout': 0.47}
        #     },
        #     'optimizer': {'name': 'AdamW', 'params': {'lr': 1e-4, 'weight_decay': 1e-5}},
        #     'scheduler': {'name': 'StepLR', 'params': {'step_size': 15, 'gamma': 0.5}}, 
        #     'train_params': {'num_epochs': 60, 'patience': 7}
        # },
    ]

    # --- Run Experiments ---
    results = []
    for config in experiment_configs:
        try:
            result = run_experiment(config, n_neurons, train_loader, val_loader, device)
            results.append(result)
        except Exception as e:
            import traceback
            print(f"!!!!! Experiment {config.get('name', 'Unnamed')} failed: {e} !!!!!")
            traceback.print_exc() # Print full traceback for debugging
            results.append({
                'config_name': config.get('name', 'Unnamed'),
                'architecture': config.get('model_config', {}).get('architecture', 'unknown'),
                'best_val_ev': -float('inf'), 
                'error': str(e)
            })

    # --- Save and Display Results ---
    print("\n===== Experiment Summary =====")
    if not results:
        print("No experiments were completed successfully.")
        return
    
    results_df = pd.DataFrame(results)
    results_filename = "experiment_results.csv" 
    file_exists = os.path.exists(results_filename)
    
    try:
        # Save results
        mode = 'a' if file_exists else 'w'  # append if file exists otherwise write
        header = not file_exists            # write header only if file does not exist
        print(f"{'Appending' if file_exists else 'Creating'} results file: {results_filename}")
        results_df.to_csv(results_filename, mode=mode, header=header, index=False)
        print(f"Results saved successfully.")

        # Read all results for display
        if file_exists:  # Only try to read if file exists
            all_results_df = pd.read_csv(results_filename)
            all_results_df = all_results_df.sort_values(by='best_val_ev', ascending=False)
            
            print("\n--- Top 5 Overall Results ---")
            cols_to_print = [col for col in ['config_name', 'architecture', 'best_val_ev', 'final_val_ev', 'duration_s', 'seed'] 
                            if col in all_results_df.columns]
            print(all_results_df[cols_to_print].head(5))
            
            # Split by architecture for comparison
            print("\n--- Top Results by Architecture ---")
            for arch in all_results_df['architecture'].unique():
                arch_df = all_results_df[all_results_df['architecture'] == arch].sort_values(by='best_val_ev', ascending=False)
                print(f"\nTop 3 {arch.upper()} Models:")
                print(arch_df[cols_to_print].head(3))

    except Exception as e:
        print(f"\nError saving or reading results CSV: {e}")


if __name__ == '__main__':
     main()
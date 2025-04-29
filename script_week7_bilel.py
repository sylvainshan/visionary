#!/usr/bin/env python

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score
from torchvision import transforms
from src.utils import load_it_data


# Dataset

class ITDataset(data.Dataset):
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
            img = np.transpose(img_np, (1,2,0))
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        else:
            img = torch.tensor(img_np, dtype=torch.float32)
        return img, torch.tensor(label, dtype=torch.float32)


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
        for images, labels in train_loader:
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
            for images, labels in train_loader:
                images = images.to(device)
                out = model(images).cpu().numpy()
                preds_train.append(out)
                targs_train.append(labels.numpy())
            preds_train = np.concatenate(preds_train)
            targs_train = np.concatenate(targs_train)
            train_loss = ((preds_train - targs_train)**2).mean()
            train_ev = explained_variance_score(targs_train, preds_train)

            preds_val, targs_val = [], []
            for images, labels in val_loader:
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
    preds, targs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds.append(model(images).cpu().numpy())
            targs.append(labels.numpy())
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    print("Final R²:", r2_score(targs, preds))
    print("Final EV:", explained_variance_score(targs, preds))
    return preds, targs


# Plotting

def save_training_plots(metrics, folder="plots_cnn"):
    os.makedirs(folder, exist_ok=True)
    epochs = np.arange(1, len(metrics['train_loss'])+1)
    plt.figure(); plt.plot(epochs, metrics['train_loss'], label='Train Loss'); plt.plot(epochs, metrics['val_loss'], label='Val Loss'); plt.legend(); plt.savefig(os.path.join(folder,'loss_curve.png')); plt.close()
    plt.figure(); plt.plot(epochs, metrics['train_ev'], label='Train EV'); plt.plot(epochs, metrics['val_ev'], label='Val EV'); plt.legend(); plt.savefig(os.path.join(folder,'ev_curve.png')); plt.close()


# Main

def main():
    data_path = ''
    stim_train, stim_val, *_ , spikes_train, spikes_val = load_it_data(data_path)
    _, _, H, _ = stim_train.shape
    _, n_neurons = spikes_train.shape

    train_ds = ITDataset(stim_train, spikes_train)
    val_ds   = ITDataset(stim_val,   spikes_val)
    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = data.DataLoader(val_ds,   batch_size=32, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ShallowCNN(n_neurons).to(device)

    model, metrics = train_model(model, train_loader, val_loader, device,
                                 num_epochs=40, learning_rate=1e-3)
    preds, targs = evaluate_model(model, val_loader, device)
    save_training_plots(metrics)

if __name__ == '__main__':
    main()



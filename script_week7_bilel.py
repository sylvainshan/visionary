#!/usr/bin/env python

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, explained_variance_score
from src.utils import load_it_data


#Dataset

class ITDataset(data.Dataset):
    def __init__(self, stimuli, neural_responses, transform=None):
        
        self.stimuli = stimuli
        self.neural_responses = neural_responses
        self.transform = transform

    def __len__(self):
        return self.stimuli.shape[0]

    def __getitem__(self, idx):
        image = self.stimuli[idx]
        label = self.neural_responses[idx]
        if self.transform:
            image = self.transform(image)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

#CNN

class ShallowCNN(nn.Module):
    def __init__(self, num_neurons, input_img_size):
       
        super(ShallowCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            # Second conv layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            
            # Third conv layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  
        )

        final_size = input_img_size // 8  # each pool divides by 2 (2^3 = 8)
        flat_features = 128 * final_size * final_size
        # Fully connected layer to predict the neural responses
        self.fc = nn.Linear(flat_features, num_neurons)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x

#Training and Evaluation

def train_model(model, train_loader, val_loader, device, num_epochs=20, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    
    train_losses = []
    val_losses = []
    train_evs = []  # Explained variance on training set
    val_evs = []    # Explained variance on validation set

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Evaluate on the full training set to compute explained variance
        model.eval()
        train_predictions = []
        train_targets = []
        with torch.no_grad():
            for images, labels in train_loader:
                images = images.to(device)
                outputs = model(images)
                train_predictions.append(outputs.cpu().numpy())
                train_targets.append(labels.cpu().numpy())
        train_predictions = np.concatenate(train_predictions, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        epoch_train_ev = explained_variance_score(train_targets, train_predictions)
        train_evs.append(epoch_train_ev)

        # Validation 
        model.eval()
        running_val_loss = 0.0
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        epoch_val_ev = explained_variance_score(val_targets, val_predictions)
        val_evs.append(epoch_val_ev)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, "
              f"Train EV: {epoch_train_ev:.4f}, Val EV: {epoch_val_ev:.4f}")

    metrics = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_ev": train_evs,
        "val_ev": val_evs
    }
    return model, metrics

def evaluate_model(model, data_loader, device):
    """Evaluate the model on a given DataLoader and return predictions and metrics."""
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)

    r2 = r2_score(actuals, predictions)
    ev = explained_variance_score(actuals, predictions)
    print("Final Evaluation on Validation Data:")
    print(f"R²: {r2:.4f}")
    print(f"Explained Variance: {ev:.4f}")

    return predictions, actuals

def plot_explained_variance_per_neuron(actuals, predictions, save_path):
    """Plot the explained variance for each neuron across the validation set and save to file."""
    n_neurons = actuals.shape[1]
    per_neuron_ev = [explained_variance_score(actuals[:, i], predictions[:, i]) for i in range(n_neurons)]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_neurons), per_neuron_ev)
    plt.xlabel("Neuron Index")
    plt.ylabel("Explained Variance")
    plt.title("Explained Variance for Each IT Neuron (Validation Data)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def save_training_metrics_plots(metrics, save_folder):
    """Save plots for training/validation loss and explained variance vs. epoch."""
    epochs = np.arange(1, len(metrics["train_loss"]) + 1)
    
    # Plot Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_loss"], label="Train Loss")
    plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss vs. Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "loss_curve.png"))
    plt.close()

    # Plot Explained Variance curves
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metrics["train_ev"], label="Train Explained Variance")
    plt.plot(epochs, metrics["val_ev"], label="Validation Explained Variance")
    plt.xlabel("Epoch")
    plt.ylabel("Explained Variance")
    plt.title("Training and Validation Explained Variance vs. Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "ev_curve.png"))
    plt.close()

#Main

def main():
    #Assume the data file IT_data.h5 is in the current folder.
    data_path = ''
    (stimulus_train, stimulus_val, stimulus_test,
     objects_train, objects_val, objects_test,
     spikes_train, spikes_val) = load_it_data(data_path)

    # Get image and response dimensions
    n_stimulus, n_channels, img_size, _ = stimulus_train.shape
    _, n_neurons = spikes_train.shape
    print(f"Training on {n_stimulus} stimuli and predicting {n_neurons} neurons.")

    train_dataset = ITDataset(stimulus_train, spikes_train)
    val_dataset = ITDataset(stimulus_val, spikes_val)
    
    batch_size = 32
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = ShallowCNN(num_neurons=n_neurons, input_img_size=img_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 60
    learning_rate = 1e-6
    model, metrics = train_model(model, train_loader, val_loader, device,
                                 num_epochs=num_epochs, learning_rate=learning_rate)

    predictions, actuals = evaluate_model(model, val_loader, device)

    # Create folder for saving plots
    plots_folder = "plots_cnn"
    os.makedirs(plots_folder, exist_ok=True)

    plot_explained_variance_per_neuron(actuals, predictions, os.path.join(plots_folder, "per_neuron_ev.png"))
    
    save_training_metrics_plots(metrics, plots_folder)
    
if __name__ == '__main__':
    main()


### Utils
import h5py
import os
import pickle
import re
import csv
import glob
import torch
import gdown
import random
import gc

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import pearsonr, rankdata, spearmanr


# =============   Util Code for PART 1 - Linear + Task-Driven =============

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

# =============   Util Code for PART 2 - Shallow CNN Data-Driven =============
class ITDataset(Dataset):
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

# ============ General Utility Functions ================
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

def download_it_data(path_to_data):
    os.makedirs(path_to_data, exist_ok=True)
    output = os.path.join(path_to_data,"IT_data.h5")
    if not os.path.exists(output):
        url = "https://drive.google.com/file/d/1s6caFNRpyR9m7ZM6XEv_e8mcXT3_PnHS/view?usp=share_link"
        gdown.download(url, output, quiet=False, fuzzy=True)
    else:
        print("File already exists. Skipping download.")

def print_data_info(stimulus_train, spikes_train):
    n_stimulus, n_channels, img_size, _ = stimulus_train.shape
    n_bins , n_neurons = spikes_train.shape
    print('The train dataset contains {} stimuli and {} IT neurons'.format(n_stimulus,n_neurons))
    print('Each stimulus have {} channels (RGB)'.format(n_channels))
    print('The size of the image is {}x{}'.format(img_size,img_size))
    print('The number of bins of neuron spiking rate is {}'.format(n_bins))

def compute_metrics(y_true, y_pred,data_type):
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

""" def encode_object_base_labels(objects_train):
    ""
    Groups object labels by their alphabetic base name (e.g., 'car1', 'car2' → 'car'),
    then assigns an integer label to each base category.
    
    Args:
        objects_train (list or array): List of object strings like 'car1', 'banana4', 'dog2'
        
    Returns:
        label_dict (dict): Mapping from base name (e.g. 'car') to integer label
        object_labels (np.array): Array of integer labels corresponding to input
    ""
    # Extract base names using regex (strip digits from end)
    base_names = [re.match(r'[a-zA-Z]+', obj).group() for obj in objects_train]
    
    # Get sorted unique base names
    unique_bases = sorted(set(base_names))
    
    # Map base name to integer label
    label_dict = {base: idx for idx, base in enumerate(unique_bases)}
    
    # Assign labels
    object_labels = np.array([label_dict[base] for base in base_names])
    
    return label_dict, object_labels """

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

def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_data(augmented: bool):
    # Download and load IT data
    path_to_data = 'data'
    download_it_data(path_to_data)
    
    if augmented:
        stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = augment_data()
    else:
        stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

    # List unique training objects
    unique = list(set(objects_train))
    print("Unique objects in training set:", unique)
    print("Number of unique objects:", len(unique))

    # Print data info
    print_data_info(stimulus_train, spikes_train)

    return stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val

def augment_data():
    print("Use augmented data input")
    
# ============= Plots Functions ==============
def set_plot_color(model_type):
    """
    Set color for plots based on model type.
    """
    if 'linear' in model_type or 'ridge' in model_type:
        return 'blue'
    elif 'task' in model_type:
        return 'red'
    elif 'data' in model_type:
        return 'violet'
    elif 'best' in model_type:
        return 'green'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def plot_neurons_metrics(y_val, y_pred):
    """
    Plot the correlation and explained variance for each neuron in a single figure.
    """
    # Set font to Arial and size 16
    plt.rcParams['font.size'] = 16
    correlations = np.array([np.corrcoef(y_val[:, i], y_pred[:, i])[0, 1] for i in range(y_val.shape[1])])
    explained_variance = np.array([explained_variance_score(y_val[:, i], y_pred[:, i]) for i in range(y_val.shape[1])])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Correlation Scatter Plot
    axes[0].scatter(y_val.flatten(), y_pred.flatten(), alpha=0.5)
    axes[0].plot([-1, 1], [-1, 1], linestyle='--', color='red')  # Diagonal line for slope 1
    axes[0].set_xlabel("True Neural Activity")
    axes[0].set_ylabel("Predicted Neural Activity")
    axes[0].set_title("Correlation of Predicted and True Neural Activity")
    
    # Explained Variance Plot
    axes[1].bar(range(y_val.shape[1]), explained_variance)
    axes[1].set_xlabel("Neuron Index")
    axes[1].set_ylabel("Explained Variance")
    axes[1].set_title("Explained Variance of Predicted Neural Activity")
    
    plt.tight_layout()
    plt.show()

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


def plot_corr_ev_distribution2(r_values, ev_values, model_name): 
    """Plot the distribution of Pearson correlation coefficients and explained variance scores for all neurons."""

    # Set font size and style
    plt.rcParams['font.size'] = 14
    color = set_plot_color(model_name)
    fig, ax = plt.subplots(2, 2, figsize=(7, 3), gridspec_kw={'height_ratios': [1, 3]})

    # --- Correlation Coefficient Boxplot ---
    sns.boxplot(x=r_values, color=color, ax=ax[0, 0], linecolor='black', linewidth=1, width=0.5)
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("")
    ax[0, 0].set_xlim(0, 1)  # Match histogram limits
    ax[0, 0].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for spine in ax[0, 0].spines.values():
        spine.set_visible(False)

    # --- Correlation Coefficient Histogram ---
    sns.histplot(r_values, color=color, ax=ax[1, 0])
    ax[1, 0].set_xlim(0, 1)
    ax[1, 0].set_ylim(0, 50)
    #ax[1, 0].set_ylabel("")
    ax[1, 0].spines['top'].set_visible(False)
    ax[1, 0].spines['right'].set_visible(False)

    # --- Explained Variance Boxplot ---
    sns.boxplot(x=ev_values, color=color, ax=ax[0, 1], linecolor='black', linewidth=1, width=0.5)
    ax[0, 1].set_xlabel("")
    ax[0, 1].set_ylabel("")
    ax[0, 1].set_xlim(0, 1)
    ax[0, 1].tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
    for spine in ax[0, 1].spines.values():
        spine.set_visible(False)

    # --- Explained Variance Histogram ---
    sns.histplot(ev_values, color=color, ax=ax[1, 1])
    ax[1, 1].set_xlim(0, 1)
    ax[1, 1].set_ylim(0, 50)
    #ax[1, 1].set_ylabel("")
    ax[1, 1].spines['top'].set_visible(False)
    ax[1, 1].spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs("out/corr_exp_var_histogram", exist_ok=True)
    outpath = f"out/corr_exp_var_histogram/{model_name}_hist.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved correlation and explained variance histograms to: {outpath}")

def plot_layer_comparison(results, n_components, save=False, path=None):
    """Compare pretrained vs random distributions for each layer's metrics using boxplots and histograms.
    
    Args:
        results (dict): Nested dictionary containing 'pearson' and 'explained_variance'
                        for 'pretrained' and 'random' models, organized by layer.
    """
    for layer in results:
        # Create figure with 2x2 grid (2 metrics x 2 plot types)
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), 
                                gridspec_kw={'height_ratios': [1, 3]})
        fig.suptitle(f'{layer} layer', y=1.02, fontsize=20)
        
        pearson_pre = results[layer]['pearson']['pretrained']
        pearson_rand = results[layer]['pearson']['random']
        ev_pre = results[layer]['explained_variance']['pretrained']
        ev_rand = results[layer]['explained_variance']['random']
        
        df_pearson = pd.DataFrame({
            'Model': ['Pretrained']*len(pearson_pre) + ['Random']*len(pearson_rand),
            'Value': np.concatenate([pearson_pre, pearson_rand])
        })
        
        df_ev = pd.DataFrame({
            'Model': ['Pretrained']*len(ev_pre) + ['Random']*len(ev_rand),
            'Value': np.concatenate([ev_pre, ev_rand])
        })
        
        mean_marker_props = {
            'marker': '^',
            'markerfacecolor': 'white',
            'markeredgecolor': 'black',
            'markersize': 8,
            'markeredgewidth': 1.5,
        }
        # Pearson plots
        sns.boxplot(data=df_pearson, x='Value', y='Model', ax=axes[0,0], orient='h', hue='Model',
                    palette=['#66b3ff', '#ff9999'], width=0.5, linecolor='black', showmeans=True, meanprops=mean_marker_props)
        axes[0,0].set_title('Correlation Coefficient Distribution', fontsize=18)
        axes[0,0].set_xlabel('')
        axes[0,0].set_ylabel('')
        axes[0,0].grid(True, alpha=0.3, linestyle='--')
        
        sns.histplot(data=df_pearson, x='Value', hue='Model', ax=axes[1,0], 
                    palette=['#66b3ff', '#ff9999'], alpha=0.5, bins=20)
        axes[1,0].set_xlabel('Correlation Coefficient', fontsize=18)
        axes[1,0].set_ylabel('Frequency', fontsize=18)
        axes[1,0].grid(True, alpha=0.3, linestyle='--')
        
        # Explained variance plots
        sns.boxplot(data=df_ev, x='Value', y='Model', ax=axes[0,1], orient='h', hue='Model',
                    palette=['#66b3ff', '#ff9999'], width=0.5, linecolor='black', showmeans=True, meanprops=mean_marker_props)
        axes[0,1].set_title('Explained Variance Distribution', fontsize=18)
        axes[0,1].set_xlabel('')
        axes[0,1].set_ylabel('')
        axes[0,1].grid(True, alpha=0.3, linestyle='--')
        
        sns.histplot(data=df_ev, x='Value', hue='Model', ax=axes[1,1], 
                    palette=['#66b3ff', '#ff9999'], alpha=0.5, bins=20)
        axes[1,1].set_xlabel('Explained Variance', fontsize=18)
        axes[1,1].set_ylabel('Frequency', fontsize=18)
        axes[1,1].grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(path, f'/{n_components}pcs/{layer}_comparison.pdf'), bbox_inches='tight')

        plt.show()

def plot_population_rdm_analysis(predicted_responses, true_responses, object_labels, model_name, metric='correlation'):
    """
    Computes and plots rank-normalized RDMs for predicted and true responses separately,
    using intelligent group labels for tick display (e.g., 'car', 'fruit', 'animal').

    Saves one plot per RDM to out/rdm/.
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

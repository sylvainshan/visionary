import os
import glob
import torch
import pickle
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet50, ResNet50_Weights


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prediction_evaluation(y_true, y_pred):
    """
    Evaluate the prediction performance of a model using Pearson correlation and explained variance for each neuron.
    
    Args:
        y_true (numpy.ndarray): True target values.
        y_pred (numpy.ndarray): Predicted values.
    
    Returns:
        dict: A dictionary containing the Pearson correlation coefficient and explained variance score.
    """
    n_neurons = y_true.shape[1]
    
    # Calculate Pearson correlation for each neuron
    r_values = np.zeros(n_neurons)
    for i in range(n_neurons):
        r, _ = pearsonr(y_true[:, i], y_pred[:, i])
        r_values[i] = r

    # Calculate explained variance for each neuron
    ev_values = np.zeros(n_neurons)
    for i in range(n_neurons):
        ev = explained_variance_score(y_true[:, i], y_pred[:, i])
        ev_values[i] = ev
    
    return {"pearson": r_values, "explained_variance": ev_values}


def evaluate_resnet(model_type, y_true, layers_of_interest, save_dir, save_dir_pc, data_path='data/task_driven_models'):
    """
    Evaluate the model performance using PCA and linear regression on the activations of the specified layers.

    Args:
        model_type (str): Type of the model ('pretrained' or 'random').
        y_true (numpy.ndarray): True target values.
        layers_of_interest (list): List of layer names to evaluate.
        save_dir (str): Directory where activations are saved.
        save_dir_pc (str): Directory where PCA models are saved.

    Returns:
        dict: A dictionary containing the evaluation results for each layer.
    """
    if model_type not in ["pretrained", "random"]:
        raise ValueError("Model must be 'pretrained' or 'random'")
    
    results = {}
    for layer_name in tqdm(layers_of_interest, desc=f"Processing {model_type} layers"):
        # Load PCA model
        pca_path = os.path.join(save_dir_pc, f'{layer_name}_pca_model.pkl')
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        
        # Load activations
        activation_files = glob.glob(os.path.join(save_dir, f'{layer_name}_batch_*.npy'))
        X = np.concatenate([np.load(f) for f in activation_files], axis=0)
        
        # Transform to PCs
        X_pca = pca.transform(X)
        
        # Train linear regression
        model = LinearRegression().fit(X_pca, y_true)
        y_pred = model.predict(X_pca)
        
        # Compute metrics
        metrics = prediction_evaluation(y_true, y_pred)
        avg_corr = np.nanmean(metrics["pearson"]) 
        avg_ev = np.nanmean(metrics["explained_variance"])
        results[layer_name] = {
            "pearson": avg_corr,
            "explained_variance": avg_ev,
            "per_neuron": metrics  
        }
        
        # save results
        filename = model_type + "_results.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
            
    return results


def evaluate_resnet50(model_type, stimulus_train, spikes_train, n_inputs, batch_size, layers_of_interest, n_components, skip_forward_pass=False, verbose=False, data_path='data'):
    """
    Evaluate a ResNet50 model (either pretrained or random) on the provided stimulus and spikes data. 
    This functions does the following
        - Loads the model (either pretrained or random).
        - Forwards the training data through the model to save activations for each layer.
        - Applies PCA to the activations of each layer and saves the PCA model.
        - Linear regression is performed on the PCA-transformed activations to predict the spikes.
        - The performance of the model is evaluated using Pearson correlation and explained variance metrics.

    Args:
        model_type (str): Type of the model to evaluate ('pretrained' or 'random').
        stimulus_train (numpy.ndarray): Training stimulus data.
        spikes_train (numpy.ndarray): Training spikes data.
        n_inputs (int): Number of inputs to process.
        batch_size (int): Batch size for processing.
        layers_of_interest (list): List of layer names to evaluate.
        n_components (int): Number of PCA components to retain.
        verbose (bool): If True, print additional information.

    Returns:
        dict: A dictionary containing the evaluation results for each layer.
    """
    if model_type == "pretrained":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
        save_dir = data_path + "/pretrained_activations"
        save_dir_pc = data_path + f"/{n_components}pcs/pretrained_pca"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_pc, exist_ok=True)
    elif model_type == "random":
        model = resnet50(weights=None).to(device)
        save_dir = data_path + "/random_activations"
        save_dir_pc = data_path + f"{n_components}pcs/random_pca"
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_dir_pc, exist_ok=True)
    else: 
        raise ValueError("Model must be 'pretrained' or 'random'")
    
    model.eval()

    if verbose: 
        print(model)

    # Prepare a dictionary to track batch indices for each layer file
    batch_counters = {layer: 0 for layer in layers_of_interest}

    def save_activation(layer_name, activation):
        # Convert tensor to numpy and save to disk as a file.
        batch_idx = batch_counters[layer_name]
        file_path = os.path.join(save_dir, f'{layer_name}_batch_{batch_idx}.npy')
        if not os.path.exists(file_path): 
            np.save(file_path, activation.cpu().numpy())
        batch_counters[layer_name] += 1

    # Define a hook that saves the activations per batch
    def get_activation_saver(layer_name):
        def hook(module, input, output):
            # Reshape activations to [batch_size, -1]
            acts = output.detach().view(output.size(0), -1)
            save_activation(layer_name, acts)
        return hook

    # Register hooks for each layer
    hooks = {}
    modules = dict(model.named_modules())
    for layer_name in layers_of_interest:
        layer = modules[layer_name]
        hooks[layer_name] = layer.register_forward_hook(get_activation_saver(layer_name))

    train_inputs = torch.tensor(stimulus_train[:n_inputs]).to(device)
    train_outputs = torch.tensor(spikes_train[:n_inputs]).to(device)
    dataset = TensorDataset(train_inputs, train_outputs)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Forward pass to save activations to disk
    if skip_forward_pass: 
        for batch_inputs, _ in tqdm(dataloader, desc="Forward pass"):
            batch_inputs = batch_inputs.to(device)
            with torch.no_grad():
                model(batch_inputs)

    # Remove hooks after processing
    for hook in hooks.values():
        hook.remove()

    # For each layer, load all corresponding files concatenate them and perform PCA
    for layer_name in tqdm(layers_of_interest, desc="PCA on layers"):
        files = glob.glob(os.path.join(save_dir, f'{layer_name}_batch_*.npy'))
        
        all_activations = []
        for file in tqdm(files, desc=f"Loading {layer_name} activations"):
            acts = np.load(file)
            all_activations.append(acts)
        layer_activations = np.concatenate(all_activations, axis=0)
        
        # Perform PCA and save the model
        pca = PCA(n_components=n_components)
        pca.fit(layer_activations)
        pca_model_path = os.path.join(save_dir_pc, f'{layer_name}_pca_model.pkl')
        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca, f)
        
        if verbose:
            print(f'PCA for {layer_name} computed with shape: {pca.components_.shape}\n')

    y_true = spikes_train[:n_inputs]
    results = evaluate_resnet(model_type, y_true, layers_of_interest, save_dir, save_dir_pc)
    
    return results


def visualize_activations(layer_name, save_dir, num_batches=5):
    # Load activations from the first few batches
    activations = []
    for batch_idx in range(num_batches):
        file_path = os.path.join(save_dir, f'{layer_name}_batch_{batch_idx}.npy')
        if os.path.exists(file_path):
            acts = np.load(file_path)
            activations.append(acts)
        else:
            print(f"File {file_path} does not exist. Skipping.")
    
    # Concatenate all activations
    all_activations = np.concatenate(activations, axis=0)
    print(f"Shape of activations for {layer_name}: {all_activations.shape}")
    
    # Define expected shapes for each layer
    layer_shapes = {
        'conv1': (64, 112, 112),
        'layer1': (256, 56, 56),
        'layer2': (512, 28, 28),
        'layer3': (1024, 14, 14),
        'layer4': (2048, 7, 7),
        'avgpool': (2048, 1, 1)
    }
    
    # Get expected shape
    expected_shape = layer_shapes.get(layer_name)
    if expected_shape is None:
        print(f"Layer {layer_name} not found in layer_shapes. Skipping visualization.")
        return
    
    # Reshape activations to original shape
    activations_reshaped = all_activations.reshape(-1, *expected_shape)
    
    # Plot activations
    plt.figure(figsize=(15, 10))
    num_samples = min(5, activations_reshaped.shape[0])
    num_channels = min(10, expected_shape[0])
    
    for sample_idx in range(num_samples):
        for channel_idx in range(num_channels):
            plt.subplot(num_samples, num_channels, sample_idx * num_channels + channel_idx + 1)
            activation = activations_reshaped[sample_idx, channel_idx]
            plt.imshow(activation, cmap='gray')
            plt.title(f'Sample {sample_idx+1}, Ch {channel_idx+1}')
            plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_corr_ev_by_neuron(r_values, ev_values): 
    """Plot the Pearson correlation coefficients and explained variance scores w.r.t to the neuron index.
    
    Args:
        r_values (numpy.ndarray): Array of Pearson correlation coefficients for each neuron.
        ev_values (numpy.ndarray): Array of explained variance scores for each neuron.
    """
    fig, ax = plt.subplots(1, 2)
    neuron_idx = range(1, len(r_values)+1)
    ax[0].plot(neuron_idx, r_values, 'o', linestyle='-')
    ax[0].set_xlabel('Neuron index')
    ax[0].set_ylabel('Correlation coefficient')
    ax[0].set_ylim(-1, 1)
    ax[0].set_xlim(0, 170)
    ax[0].grid(True, alpha=0.5, linestyle='--')
    ax[1].plot(neuron_idx, ev_values, 'o', linestyle='-', color='coral')
    ax[1].set_xlabel('Neuron index')
    ax[1].set_ylabel('Explained variance')
    ax[1].set_ylim(-1, 1)
    ax[1].set_xlim(0, 170)
    ax[1].grid(True, alpha=0.5, linestyle='--')
    plt.tight_layout()
    plt.show()


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

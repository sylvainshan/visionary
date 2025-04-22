import argparse
import time
import torch
import pickle
import random
import gc
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.decomposition import IncrementalPCA



from utils import *

# === Utils ===
# to add here for submission

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
    
# === Model Function Definitions ===

# PART 1a
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
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred)
    log_metrics_in_csv(model_name="linear",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,fig_name='linear')

def run_linear_pca(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Linear Regression with PCA...")

    # Compute PCA
    X_train, X_val = get_pca(stimulus_train.reshape(stimulus_train.shape[0], -1), stimulus_val.reshape(stimulus_val.shape[0], -1), n_components=1000)

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
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred)
    log_metrics_in_csv(model_name="linear_pca",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,fig_name='linear_pca')

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
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred)
    log_metrics_in_csv(model_name="ridge_cv5",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,fig_name='ridge_cv5')

def run_ridge_pca_cv5(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Ridge Regression with PCA and 5-fold CV...")

    # Compute PCA
    X_train, X_val = get_pca(stimulus_train.reshape(stimulus_train.shape[0], -1), stimulus_val.reshape(stimulus_val.shape[0], -1), n_components=1000)

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
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred)
    log_metrics_in_csv(model_name="ridge_pca_cv5",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,fig_name='ridge_pca_cv5')
    
def run_elasticnet_pca_cv5(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):

    print("Running Elastic Net Regression with PCA and 5-fold CV...")

    # Compute PCA
    X_train, X_val = get_pca(
        stimulus_train.reshape(stimulus_train.shape[0], -1),
        stimulus_val.reshape(stimulus_val.shape[0], -1),
        n_components=1000
    )

    # Elastic Net setup
    alphas = np.linspace(160, 200, 10)  # Regularization strength
    l1_ratios = np.logspace(-10, -8, 10)  # Mix between L1 (Lasso) and L2 (Ridge)

    elastic_net = ElasticNet()

    # Stratified K-Fold using class labels
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search with stratified CV
    grid = GridSearchCV(
        estimator=elastic_net,
        param_grid={
            'alpha': alphas,
            'l1_ratio': l1_ratios
        },
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
    best_params = grid.best_params_
    print(f"\nBest parameters: alpha={best_params['alpha']}, l1_ratio={best_params['l1_ratio']}")
    print(f"Best CV MSE: {-grid.best_score_:.4f}")

    # Train best model on full training set
    model = ElasticNet(alpha=best_params['alpha'], l1_ratio=best_params['l1_ratio'])
    start_time = time.time()
    model.fit(X_train, spikes_train)
    computation_time = time.time() - start_time
    print(f"Done. Training time: {computation_time:.4f}")

    # Evaluate on validation set
    spikes_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(spikes_val, spikes_val_pred)
    print(f"Validation MSE with best Elastic Net model: {val_mse:.4f}")

    # Compute and log metrics
    r2, mse, mae, mape, ev_val, ev_avg, corr_val, corr_avg = compute_metrics(spikes_val, spikes_val_pred)
    log_metrics_in_csv(
        model_name="elasticnet_pca_cv5",
        augmented=augmented,
        r2=r2,
        mse=mse,
        mae=mae,
        mape=mape,
        ev_avg=ev_avg,
        corr_avg=corr_avg,
        time_comput=computation_time
    )

    # Plot correlation and EV distributions
    plot_corr_ev_distribution(corr_val, ev_val, fig_name='elasticnet_pca_cv5')

# PART 1b   
def run_task_driven(model,device,stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented,model_type,verbose=True):
    print(f"Running Task-driven model ({model_type} weights) using {augmented} dataset...")
    
    # === Parameters ====
    DATA_PATH =f'data/task_driven_models'
    BATCH_SIZE = 100
    N_PCA = 1000
    N_INPUTS = len(stimulus_train)
    # ===================

    layers_of_interest = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

     # Create output directories for saving activations and PCA models
    save_dir = os.path.join(DATA_PATH,f"{model_type}_activations")
    save_dir_pc =  os.path.join(DATA_PATH,f"{N_PCA}pcs",f"{model_type}_pca")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_pc, exist_ok=True)

    # Set model to evaluation mode
    model.eval()

    # Track how many batches are saved per layer
    batch_counters = {layer: 0 for layer in layers_of_interest}

    # ===== Set hooks ===================================================================================
    # Save activations
    def save_activation(layer_name, activation, data_type):
        # Convert tensor to numpy and save to disk as a file.
        batch_idx = batch_counters[layer_name]
        file_path = os.path.join(save_dir, f'{layer_name}_{data_type}_batch_{batch_idx}.npy')
        if not os.path.exists(file_path): 
            np.save(file_path, activation.cpu().numpy())
        batch_counters[layer_name] += 1

    # Hook generator to capture and save activations during forward pass
    def get_activation_saver(layer_name, data_type):
        def hook(module, input, output):
            # Reshape activations to [batch_size, -1]
            acts = output.detach().view(output.size(0), -1) # [batch_size, features]
            save_activation(layer_name, acts, data_type)
        return hook
    
    def register_hooks(model, data_type):
        hooks = {}
        for layer_name in layers_of_interest:
            layer = modules[layer_name]
            hooks[layer_name] = layer.register_forward_hook(get_activation_saver(layer_name, data_type))
        return hooks

    # ===== Prepare Dataloader - Run data through model to trigger hooks and save activations =========================
    modules = dict(model.named_modules())  # define this once at the top

    # Iterate Train and Validation dataset
    for data_type, stimulus, spikes in zip(['train','val'],[stimulus_train,stimulus_val],[spikes_train,spikes_val]):

        # Prepare data loader
        inputs = torch.tensor(stimulus).to(device)
        outputs = torch.tensor(spikes).to(device)
        dataset = TensorDataset(inputs, outputs)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        
        hooks = register_hooks(model, data_type)
        for batch_x, _ in dataloader:
            _ = model(batch_x)
        for hook in hooks.values():
            hook.remove()

        #  ===== Apply PCA on saved activations of trained dataset for each layer ==============================================
        if data_type == 'train':
            print("Compute PCA on each layer...")
            for layer_name in layers_of_interest:
                files = glob.glob(os.path.join(save_dir, f'{layer_name}_{data_type}_batch_*.npy'))
                
                all_activations = []
                for file in tqdm(files, desc=f"Loading {layer_name} {data_type} activations"):
                    acts = np.load(file)
                    all_activations.append(acts.reshape(acts.shape[0], -1))
                layer_activations = np.concatenate(all_activations, axis=0)

                # Fit PCA and save the PCA model
                pca = PCA(n_components=N_PCA)
                pca.fit(layer_activations)
                pca_model_path = os.path.join(save_dir_pc, f'{layer_name}_{data_type}_pca_model.pkl')
                with open(pca_model_path, 'wb') as f:
                    pickle.dump(pca, f)
                
                if verbose:
                    print(f'PCA for {layer_name} {data_type} computed with shape: {pca.components_.shape}\n')
        
    # ======== Evaluate the taks-driven model using the PCA-reduced representations ======== 
    print("Apply PCs to activations and fit a linear Ridge regression model to predict brain activity")
    for layer_name in layers_of_interest:
        print(f"\nPredict: {layer_name}\n===============")
        # Load PCA model for both train and validation dataset
        pca_path = os.path.join(save_dir_pc, f'{layer_name}_train_pca_model.pkl')
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        
        # ======= Load activations and transform to PCs ======= 
        print("Loading activations and fit the PCs")
        # for train dataset
        train_activation_files = glob.glob(os.path.join(save_dir, f'{layer_name}_train_batch_*.npy'))
        X_activation_train = np.concatenate([np.load(f) for f in train_activation_files], axis=0)
        print("Train set (sample,activation layer features): ",X_activation_train.shape)
        X_train = pca.transform(X_activation_train)
        print("Train set (sample, PCs): ",X_train.shape)
        del train_activation_files, X_activation_train
        
        # for val dataset
        val_activation_files = glob.glob(os.path.join(save_dir, f'{layer_name}_val_batch_*.npy'))
        X_activation_val = np.concatenate([np.load(f) for f in val_activation_files], axis=0)
        print("Validation set (sample, activation layer features): ",X_activation_val.shape)
        X_val = pca.transform(X_activation_val)
        print("Validation set (sample, PCs): ",X_val.shape)
        del val_activation_files, X_activation_val

        # ======= Ridge regression setup ======= 
        print("Linear Regression using Ridge (grid search on alpha)")
        alphas = np.logspace(2,10, 9)
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
        print(f"Best alpha: {grid.best_params_['alpha']}")
        print(f"Best CV MSE: {-grid.best_score_:.4f}")

        # Train best model on full training set
        start_time = time.time()
        ridge_model = Ridge(alpha=grid.best_params_['alpha'])
        ridge_model.fit(X_train, spikes_train)
        computation_time = time.time() - start_time
        print(f"done. Computation time: {computation_time:.4f}")

        # Evaluate on validation set (optional)
        spikes_val_pred = ridge_model.predict(X_val)
        spikes_train_pred = ridge_model.predict(X_train)
        val_mse = mean_squared_error(spikes_val, spikes_val_pred)
        print(f"Validation MSE with best Ridge model: {val_mse:.4f}")

        # Compute metrics for training and validation set
        (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_train, spikes_train_pred)
        (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred)
        log_metrics_in_csv(model_name=f'task_driven_{model_type}_{layer_name}', augmented=augmented, r2=r2_val, mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

        # plot correlation and explained-variance distribution
        plot_corr_ev_distribution(corr_val, ev_val, fig_name=f'task_driven_{model_type}_{layer_name}')

def run_task_driven_pretrained(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    run_task_driven(model,device,stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented,model_type='pretrained')

def run_task_driven_random(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=None).to(device)
    run_task_driven(model,device,stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented,model_type='random')

# PART 2
def run_data_driven(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Data-driven model...")

# PART 3
def run_best(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running Best model (as selected by validation performance)...")

def run_all(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    print("Running All models... ")

# === Parser Setup ===
def parse_args():
    parser = argparse.ArgumentParser(description="NX-414 - Predicting Neural Activity")

    parser.add_argument(
        "--model",
        choices=[
            "linear",
            "linear_pca",
            "ridge_cv5",
            "ridge_pca_cv5",
            "enet_pca_cv5",
            "task_driven_pretrained",
            "task_driven_random",
            "data_driven",
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
            "  best                 - Best-performing model\n"
            "  all                  - Run all models sequentially and output metrics as CSV"
        )
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use augmented dataset instead of original"
    )

    return parser.parse_args()

# === Main Execution ===
if __name__ == "__main__":
    args = parse_args()

    # Fix seed to 42
    seed_everything(42)

    # Load data and compute PCA
    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = get_data(augmented=args.augment)

    # run selected model
    model_map = {
        "linear": run_linear,
        "linear_pca": run_linear_pca,
        "ridge_cv5": run_ridge_cv5,
        "ridge_pca_cv5": run_ridge_pca_cv5,
        "enet_pca_cv5": run_elasticnet_pca_cv5,
        "task_driven_pretrained": run_task_driven_pretrained,
        "task_driven_random": run_task_driven_random,
        "data_driven": run_data_driven,
        "best": run_best,
        "all": run_all,
    }

    model_function = model_map.get(args.model)
    if model_function:
        model_function(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,args.augment)
    else:
        print("Invalid model selected.")
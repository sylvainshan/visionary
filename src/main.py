import argparse
import time
import torch
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, StratifiedKFold

import torch.utils.data as data
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import DataLoader, TensorDataset, Dataset

from utils import *

# === Utils ===
# to add here for submission

# === Run Model Function Definitions ===

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
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name="linear",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

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
    log_metrics_in_csv(model_name="linear_pca",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

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
    log_metrics_in_csv(model_name="ridge_cv5",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

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
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name="ridge_pca_cv5",augmented=augmented,r2=r2_val,mse=mse_val,mae=mae_val,mape=mape_val,ev_avg=ev_avg_val,corr_avg=corr_avg_val,time_comput=computation_time)

    # Compute metrics for training and validation set
    (r2_train, mse_train, mae_train, mape_train, ev_train, ev_avg_train, corr_train, corr_avg_train) = compute_metrics(spikes_train, spikes_train_pred,'train')
    (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_val_pred,'val')
    log_metrics_in_csv(model_name='ridge_pca_cv5', augmented=augmented, r2=r2_val, mse=mse_val, mae=mae_val, mape=mape_val, ev_avg=ev_avg_val, corr_avg=corr_avg_val, time_comput=computation_time)
    log_metrics_in_csv(model_name='ridge_pca_cv5', augmented=augmented, r2=r2_train, mse=mse_train, mae=mae_train, mape=mape_train, ev_avg=ev_avg_train, corr_avg=corr_avg_train, time_comput=computation_time)
    
    # plot correlation and explained-variance distribution
    plot_corr_ev_distribution(corr_val, ev_val,'ridge_pca_cv5')

    # plot dissimilarity matrix on validation set
    plot_population_rdm_analysis(spikes_val_pred,spikes_val, objects_val, 'ridge_pca_cv5', metric='correlation')

    # plot neuron site brain activity prediction
    for site in [39,107,152]:
        plot_neuron_site_response(spikes_val_pred,spikes_val, objects_val, site, 'ridge_pca_cv5')

# PART 1b   
def run_task_driven(model,device,stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented,model_type,verbose=True):
    print(f"=== Running Task-driven model ({model_type} weights) using {'augmented' if augmented else 'original'} dataset ===")
    
    # === Parameters ====
    DATA_PATH =f'data/task_driven_models'
    BATCH_SIZE = 100
    N_PCA = 1000
    # ===================

    layers_of_interest = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

     # Create output directories for saving activations and PCA models
    save_dir = os.path.join(DATA_PATH,f"{model_type}_activations")
    save_dir_pc =  os.path.join(DATA_PATH,f"{N_PCA}pcs",f"{model_type}_pca")
    os.makedirs(save_dir, exist_ok=True)

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

    compute_pcs = False
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
       
       # plot correlation and explained-variance distribution
        plot_corr_ev_distribution(corr_val, ev_val, f'task_driven_{model_type}_{layer_name}')

        # plot dissimilarity matrix on validation set
        plot_population_rdm_analysis(spikes_val_pred,spikes_val, objects_val, f'task_driven_{model_type}_{layer_name}', metric='correlation')

        # plot neuron site brain activity prediction
        #for site in [23,39,42,56,71,95,107,132,139,152]:
        for site in [39,107,152]:
            plot_neuron_site_response(spikes_val_pred,spikes_val, objects_val, site, f'task_driven_{model_type}_{layer_name}')

def run_task_driven_pretrained(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    run_task_driven(model,device,stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented,model_type='pretrained')

def run_task_driven_random(stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val,augmented):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet50(weights=None).to(device)
    run_task_driven(model, device, stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val, augmented,model_type='random')

# PART 2
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

# PART 3
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
    print("Running Best model (as selected by validation performance)...")
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

# === Main Execution ===
if __name__ == "__main__":
    args = parse_args()

    # Fix seed to 42
    # Set the seed globally 
    SEED = 42
    seed_everything(SEED)
    set_seed(SEED)
    
    # Load data and compute PCA
    stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = get_data(augmented=args.augment)

    # run selected model
    model_map = {
        "linear": run_linear,
        "linear_pca": run_linear_pca,
        "ridge_cv5": run_ridge_cv5,
        "ridge_pca_cv5": run_ridge_pca_cv5,
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
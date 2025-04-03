import pandas as pd
import time

from utils import *

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LassoCV, ElasticNetCV, SGDRegressor
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

import torch
import torch.nn as nn
import torch.optim as optim

# Download and load IT data
path_to_data = 'data'
download_it_data(path_to_data)
stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

# Print data info
print_data_info(stimulus_train, spikes_train)

# Compute PCA
X_train_pca, X_val_pca = get_pca(stimulus_train.reshape(stimulus_train.shape[0], -1), stimulus_val.reshape(stimulus_val.shape[0], -1), n_components=1000)

use_sklearn = True
use_torch = False
metrics = []

if use_sklearn:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
    # List Linear Models
    models = {
            #"linear_cf": LinearRegression(),
            "ridge_cf": RidgeCV(alphas=np.logspace(5,6,20)),
            #"lasso_cf": LassoCV(alphas=np.logspace(5,6,100), max_iter=10000,verbose=True),
            #"elastic_net_cf": ElasticNetCV(alphas=np.logspace(5,6,100), l1_ratio=[0.1, 0.5, 0.9], verbose=1,max_iter=10000),
            #"sgd_linear": SGDRegressor(penalty='l2',alpha=500000, max_iter=10000,tol=1e-3, verbose=True),
            #"sgd_ridge": SGDRegressor(penalty='l2', alpha=500000, max_iter=1000,),
            #"sgd_elastic_net": SGDRegressor(penalty='l1', alpha=100, max_iter=1000),
        }


    for input_type, (X_train, X_val) in zip(
        [ "raw","pca"], 
        [(stimulus_train.reshape(stimulus_train.shape[0], -1), stimulus_val.reshape(stimulus_val.shape[0], -1)), 
        (X_train_pca, X_val_pca)]):
        
        # iterate models
        for model_name, model in models.items():

            print(f"Training {model_name} with {input_type} input data... ",end='', flush=True)

            if "sgd" in model_name:
                model = RegressorChain(model)

            # Fit the model
            start_time = time.time()
            model.fit(X_train, spikes_train)
            print(model.alpha_)
            computation_time = time.time() - start_time
            print(f"done. Computation time: {computation_time:.4f}")

            # Predict on both training and validation data
            spikes_pred_train = model.predict(X_train)
            spikes_pred_val = model.predict(X_val)

            # Compute metrics for training and validation set
            (r2_train, mse_train, mae_train, mape_train, ev_train, ev_avg_train, corr_train, corr_avg_train) = compute_metrics(spikes_train, spikes_pred_train)
            (r2_val, mse_val, mae_val, mape_val, ev_val, ev_avg_val, corr_val, corr_avg_val) = compute_metrics(spikes_val, spikes_pred_val)

            # Store results in a single row
            metrics.append({
                "model": model_name,
                "input_type": input_type,
                "compute_time": computation_time,
                "r2_train": r2_train,
                "mse_train": mse_train,
                "mae_train": mae_train,
                "mape_train": mape_train,
                "ev_avg_train": ev_avg_train,
                "corr_avg_train": corr_avg_train, 
                "r2_val": r2_val,
                "mse_val": mse_val,
                "mae_val": mae_val,
                "mape_val": mape_val,
                "ev_avg_val": ev_avg_val, 
                "corr_avg_val": corr_avg_val 
            })

            # Assuming r_values and ev_values are computed elsewhere
            #plot_corr_ev_distribution(corr_train, ev_train)
            #plot_corr_ev_distribution(corr_val, ev_val)

if use_torch:

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim, bias=True)  # Linear transformation

        def forward(self, x):
            return self.linear(x)
        
    # Assuming X and y are NumPy arrays of shape (2500, 150528) and (2500, 168)
    X_tensor = torch.tensor(X_train_pca, dtype=torch.float32)  # Input features
    y_tensor = torch.tensor(spikes_train, dtype=torch.float32)  # Target outputs

    # Create a DataLoader for batch processing
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model
    input_dim = X_train_pca.shape[1]  # 150528
    output_dim = spikes_train.shape[1]  # 168
    model = LinearRegressionModel(input_dim, output_dim)

    # Define loss and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)  # Ridge (L2) Regularization

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()  # Reset gradients
            y_pred = model(batch_X)  # Forward pass
            loss = criterion(y_pred, batch_y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


    # Convert results to DataFrame print and save as csv
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv("out/linear_models/linear_models_metrics.csv", index=False)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


def linear_regression(type, X_train, y_train, X_val, y_val, alpha_values=[0.01], l1_ratio_values=[0.5]):
    """Train a linear regression model and evaluate its performance.
    
    Args:
        type (str): Type of regression ('linear', 'ridge', 'lasso', 'elastic_net').
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training target values.
        X_val (numpy.ndarray): Validation features.
        y_val (numpy.ndarray): Validation target values.
        alpha_values (list): List of alpha values for Ridge and Lasso regression.
        l1_ratio_values (list): List of L1 ratios for Elastic Net regression.

    Returns:
        dict: A dictionary containing the trained model, mean squared error, predicted values, R-squared score, and explained variance score.
    """
    if type == 'linear': 
        model = LinearRegression()
        model.fit(X_train, y_train)
   
    elif type == 'ridge' or type == 'lasso':
        param_grid = {'alpha': alpha_values}
        if type == 'ridge':
            model = Ridge()
        else:
            model = Lasso()
        # use halving random grid search to find the best alpha value
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_alpha = grid_search.fit(X_train, y_train).best_params_['alpha']
        model = grid_search.best_estimator_

    elif type == 'elastic_net':
        param_grid = {'alpha': alpha_values, 'l1_ratio': l1_ratio_values}
        model = ElasticNet()
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_alpha = grid_search.fit(X_train, y_train).best_params_['alpha']
        best_l1_ratio = grid_search.fit(X_train, y_train).best_params_['l1_ratio']
        model = grid_search.best_estimator_
    
    else: 
        raise ValueError("Invalid regression type. Choose from 'linear', 'ridge', 'lasso', or 'elastic_net'.")
    
    y_pred = model.predict(X_val)
    scores = evaluate_model(y_pred, y_val)
    return {"model": model, 
            "y_pred": y_pred, 
            "mse": scores["mse"], 
            "r2": scores["r2"], 
            "ev": scores["ev"], 
            "alpha": best_alpha if type in ['ridge', 'lasso', 'elastic_net'] else None, 
            "l1_ratio": best_l1_ratio if type == 'elastic_net' else None}


def evaluate_model(y_pred, y_val):
    """Evaluate the performance of a regression model."
    
    Args:
        y_pred (numpy.ndarray): Predicted values.
        y_val (numpy.ndarray): True target values.
    
    Returns:
        dict: A dictionary containing the mean squared error, R-squared score, and explained variance score.
    """
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    ev = explained_variance_score(y_val, y_pred)
    return {"mse": mse, "r2": r2, "ev": ev}


def compute_pca(X_train, X_val, n_components=20, verbose=False):
    """Perform PCA on the training set, transform both training and validation sets, and return the transformed data."
    
    Args:
        X_train (numpy.ndarray): Training features.
        X_val (numpy.ndarray): Validation features.
        n_components (int): Number of principal components to keep.

    Returns:
        tuple: Transformed training and validation sets.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_val = pca.transform(X_val)
    if verbose:
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_)}")
        print(f"Number of components: {pca.n_components_}")
        print(f"Original shape: {X_train.shape}")
        print(f"Reduced shape: {X_train.shape}")
    return X_train, X_val 


def print_results(model_name, results):
    """Print the evaluation results of a regression model.

    Args:
        model_name (str): Name of the regression model.
        results (dict): Dictionary containing the evaluation results.
    """
    mse = results['mse']
    r2 = results['r2']
    ev = results['ev']
    best_alpha = results.get('alpha', None)
    best_l1_ratio = results.get('l1_ratio', None)
    print(f"{model_name} results:")
    print(f"- Mean Squared Error: {mse:.4f}")
    print(f"- R-squared: {r2:.4f}")
    print(f"- Explained Variance: {ev:.4f}")
    if best_alpha is not None:
        print(f"- Best alpha: {best_alpha}")
    if best_l1_ratio is not None:
        print(f"- Best L1 ratio: {best_l1_ratio}")


def prediction_evaluation(y_true, y_pred):
    """Evaluate the prediction performance of a model using Pearson correlation and explained variance for each neuron.
    
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


def plot_neuron_evaluation(y_true, y_pred, neuron_idx):
    """Plot the Pearson correlation between true and predicted values for a specific neuron.
    
    Args:
        y_true (numpy.ndarray): True target values.
        y_pred (numpy.ndarray): Predicted values.
        neuron_idx (int): Index of the neuron to plot.
    """
    r_values, ev_values = prediction_evaluation(y_true, y_pred).values()
    
    print("Averages over neurons:")
    print(f"- Average fraction of explained variance: {np.mean(ev_values):.2f}")
    print(f"- Average Pearson correlation: {np.mean(r_values):.2f}\n")

    print(f"Evaluation for neuron {neuron_idx}:")
    print(f"- Fraction of explained variance for neuron {neuron_idx}: {ev_values[neuron_idx]:.2f}")
    print(f"- Pearson correlation for neuron {neuron_idx}: {r_values[neuron_idx]:.2f}")

    plt.figure()
    sns.regplot(x=y_true[:, neuron_idx], y=y_pred[:, neuron_idx], line_kws={"color": "red"})
    plt.title(f'Pearson correlation for Neuron {neuron_idx} (r={r_values[neuron_idx]:.2f})')
    plt.xlabel('True Spikes')
    plt.ylabel('Predicted Spikes')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    

def plot_multiple_neuron_evaluations(y_true, y_pred, neuron_indices, n_cols=5):
    """Plot the Pearson correlation between true and predicted values for multiple neurons.
    
    Args:
        y_true (numpy.ndarray): True target values.
        y_pred (numpy.ndarray): Predicted values.
        neuron_indices (list): List of neuron indices to plot.
        n_cols (int): Number of columns in the subplot grid.
    """
    r_values, _ = prediction_evaluation(y_true, y_pred).values()
    n_neurons = len(neuron_indices)
    n_rows = (n_neurons + n_cols - 1) // n_cols  
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True, sharey=True)
    axes = axes.flatten()  
    
    for i, neuron_idx in enumerate(neuron_indices):
        ax = axes[i]
        sns.regplot(x=y_true[:, neuron_idx], y=y_pred[:, neuron_idx], ax=ax, line_kws={"color": "red"})
        ax.set_title(fr'Neuron {neuron_idx} ($r={r_values[neuron_idx]:.2f}$)')
        ax.grid(True, alpha=0.5, linestyle='--')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        if i % n_cols == 0:
            ax.set_ylabel('Predicted Spikes')
        else:
            ax.set_ylabel('')
        
        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel('True Spikes')
        else:
            ax.set_xlabel('')
    
    # Delete empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_corr_ev_distribution(r_values, ev_values): 
    """Plot the distribution of Pearson correlation coefficients and explained variance scores for all neurons.

    Args:
        r_values (numpy.ndarray): Array of Pearson correlation coefficients for each neuron.
        ev_values (numpy.ndarray): Array of explained variance scores for each neuron.
    """
    fig, ax = plt.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [1, 3]}, figsize=(12,6))
    sns.boxplot(x=r_values, color='skyblue', ax=ax[0,0], linecolor='black', linewidth=1, width=0.5)
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("")
    ax[0, 0].set_title("Correlation Coefficient Distribution")
    sns.histplot(r_values, color='skyblue', ax=ax[1,0])
    ax[1,0].set_xlabel("Correlation Coefficient")
    ax[1,0].set_ylabel("Frequency")
    sns.boxplot(x=ev_values, color='coral', ax=ax[0,1], linecolor='black', linewidth=1, width=0.5)
    ax[0, 1].set_xlabel("")
    ax[0, 1].set_ylabel("")
    ax[0, 1].set_title("Explained Variance Distribution")
    sns.histplot(ev_values, color='coral', ax=ax[1,1])
    ax[1,1].set_xlabel("Explained Variance")
    ax[1,1].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_corr_ev_by_neuron(r_values, ev_values): 
    """Plot the Pearson correlation coefficients and explained variance scores w.r.t to the neuron index.
    
    Args:
        r_values (numpy.ndarray): Array of Pearson correlation coefficients for each neuron.
        ev_values (numpy.ndarray): Array of explained variance scores for each neuron.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
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
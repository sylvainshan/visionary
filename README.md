# 🧠 NX-414 - Predicting Neural Activity

This project provides a modular pipeline for training and evaluating different models on neural activity data from the IT cortex. It supports both classical regression models and modern task- or data-driven architectures.

---

## 🚀 Getting Started

Make sure all required dependencies are installed (`numpy`, `pandas`, `matplotlib`, `seaborn`, `torch`, `torchvision`, `scikit-learn`, `h5py`, `gdown`, `tqdm`, `scipy`), then run:

```bash
python main.py [--model MODEL_NAME] [--augment]
```

If no model is specified, the script defaults to `best`.

⚠️ We recommend having more than 64GB of memory to handle larger models and datasets efficiently.

---

## ⚙️ Command-Line Arguments

### `--model`

Specifies which model to run.

**Choices:**

- `linear` – Basic Linear Regression
- `linear_pca` – Linear Regression with PCA
- `ridge_cv5` – Ridge Regression with 5-fold Cross-Validation
- `ridge_pca_cv5` – Ridge Regression with PCA and CV
- `task-driven_trained` – Task-driven model using trained weights
- `task-driven_random` – Task-driven model using random weights
- `data-driven` – Data-driven shallow CNN model
- `resnet_adapter` – Data-driven shallow CNN model (Best)
- `vgg_bn` – Data-driven shallow CNN model
- `best` – Best-performing model (default)
- `all` – Run all models sequentially

**Default:** `best`

---

### `--augment`

If set, uses the augmented dataset instead of the original dataset.

```bash
--augment
```

This flag has no argument — it's either used or not used.

---

## 📆 Output

- Model performance metrics are logged to a CSV file at:

  ```
  out/metrics_log.csv
  ```

- Each row in the CSV includes:

  - `datetime` – Timestamp of the run
  - `model` – Name of the model used
  - `augment` – Whether augmentation was used (1 or 0)
  - `r2` – R² Score
  - `mse` – Mean Squared Error
  - `mae` – Mean Absolute Error
  - `mape` – Mean Absolute Percentage Error
  - `ev_avg` – Average Explained Variance
  - `corr_avg` – Average Pearson Correlation

---

## 📊 Output Figures

This project generates several plots that help analyze model performance and behavior. All figures are saved under the `out/` directory. Below is a description of the figures generated:

- Correlation and Explained Variance Distributions

  **Directory:** `out/corr_exp_var_histogram/{model_name}_hist.png`
  A two-panel plot displaying histograms and boxplots for:

  - Pearson correlation coefficients
  - Explained variance (EV) scores

  Each metric is shown as a percentage across all neurons. Helps assess model performance distributionally.

---

- Layer-wise Explained Variance
  **Directory:** `out/layer_comparison/{model_type}_layer_comparison.png` 
  Bar plot comparing explained variance across layers of a model. Useful for understanding how information is represented at different stages in task-driven architectures.

---

- Representational Dissimilarity Matrices (RDMs)
  **Directory:** `out/rdm/{model_name}_predicted_rdm.png`
  Visual comparison between predicted and ground truth RDMs. RDMs are rank-normalized and sorted by semantic categories. Helps evaluate how closely model representations align with neural population activity.

---

- Single Neuron Response Profiles

  **Directory:** `out/neuron_site/{model_name}_site{site_index}.png`  
  Line plot showing predicted vs. true neural responses for a specific neuron (site), across all stimuli. Object categories are indicated via background grouping. Great for detailed per-site inspection.


## 🧪 Example Usage

Run the best model (default behavior):

```bash
python main.py
```

Run Ridge Regression with PCA and 5-fold CV using augmented data:

```bash
python main.py --model ridge_pca_cv5 --augment
```

Run all models and log results to CSV:

```bash
python main.py --model all
```

---

## 📁 Project Structure (simplified)

```
project/
│
├── main.py                 # Entry point
├── data/                   # Raw and augmented data
└── out/
    ├── metrics_log.csv     # Metrics log file
    ├── corr_exp_var_histogram/     # Histogram plots
    ├── layer_comparison/     # Layer comparison plots
    ├── neuron_site/         # Single site prediction plots
    └── rdm/         # Representational dissimilarity matrices plots
```

## 🔒 Reproducibility

To ensure reproducible results, we set random seeds for Python, NumPy, and PyTorch. This also includes configuring the CUDA backend and DataLoader workers. By fixing these seeds and environment variables, we minimize variability across runs, though minor variations in results might still be observed.

---

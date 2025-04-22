# 🧠 NX-414 - Predicting Neural Activity

This project provides a modular pipeline for training and evaluating different models on neural activity data from the IT cortex. It supports both classical regression models and modern task- or data-driven architectures.

---

## 🚀 Getting Started

Make sure all required dependencies are installed (e.g. `numpy`, `scikit-learn`, etc.), then run:

```bash
python main.py [--model MODEL_NAME] [--augment]
```

If no model is specified, the script defaults to `best`.

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
- `data-driven` – Data-driven deep learning model
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
├── models/                 # Model definitions
├── data/                   # Raw and augmented data
├── utils/                  # Utility scripts
└── out/
    └── metrics_log.csv     # Metrics log file
```

---

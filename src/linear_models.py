from src.utils import *

# Download and load IT data
path_to_data = 'data'
download_it_data(path_to_data)
stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

# Flatten images for regression
X_train = stimulus_train.reshape(stimulus_train.shape[0], -1)
X_val = stimulus_val.reshape(stimulus_val.shape[0], -1)
y_train = spikes_train
y_val = spikes_val

# Print data info
print_data_info(stimulus_train, spikes_train)

model = select_model("ridge_cv")

# Fit the model using RidgeCV and extract the best alpha
model = model.fit(X_train, y_train)

# Predict on validation data
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
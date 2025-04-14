import os
from utils import *
from utils_task_driven_models import *

path_to_data = 'data/'
stimulus_train, stimulus_val, stimulus_test, objects_train, objects_val, objects_test, spikes_train, spikes_val = load_it_data(path_to_data)

# === Parameters ====
SAVE_FIG = True
FIG_PATH = 'Figures/task_driven_models'
DATA_PATH = 'data/task_driven_models'
RANDOM_SEED = 42
BATCH_SIZE = 100
N_PCA_COMPONENTS = 1000
N_INPUTS = len(stimulus_train)
# ===================

os.makedirs(FIG_PATH, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
seed_everything(RANDOM_SEED)
layers_of_interest = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

# Evaluate the ResNet50 pretrained model
if not os.path.exists(DATA_PATH + "/pretrained_results.pkl"):
    pretrained_results = evaluate_resnet50(model_type="pretrained", 
                            stimulus_train=stimulus_train, 
                            spikes_train=spikes_train,
                            n_inputs=N_INPUTS, 
                            batch_size=BATCH_SIZE,
                            layers_of_interest=layers_of_interest,
                            n_components=N_PCA_COMPONENTS,
                            verbose=True,
                            data_path=DATA_PATH)
else: 
    with open(DATA_PATH + "/pretrained_results.pkl", "rb") as f:
        pretrained_results = pickle.load(f)

# Evaluate the ResNet50 random model
if not os.path.exists(DATA_PATH + "/random_results.pkl"):
    random_results = evaluate_resnet50(model_type="random", 
                            stimulus_train=stimulus_train, 
                            spikes_train=spikes_train,
                            n_inputs=N_INPUTS, 
                            batch_size=BATCH_SIZE,
                            layers_of_interest=layers_of_interest,
                            n_components=N_PCA_COMPONENTS,
                            verbose=True,
                            data_path=DATA_PATH)
else:
    with open(DATA_PATH + "/random_results.pkl", "rb") as f:
        random_results = pickle.load(f)

# Aggregate results for each layer and compare pretrained vs random
results = {}
for layer in layers_of_interest:
    results[layer] = {
        "pearson": {
            "pretrained": pretrained_results[layer]["per_neuron"]["pearson"],
            "random": random_results[layer]["per_neuron"]["pearson"]
        },
        "explained_variance": {
            "pretrained": pretrained_results[layer]["per_neuron"]["explained_variance"],
            "random": random_results[layer]["per_neuron"]["explained_variance"]
        },
    }
plot_layer_comparison(results, save=True, path=FIG_PATH)
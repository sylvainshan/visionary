### Utils
import h5py
import os

import matplotlib.pyplot as plt
import numpy as np

import gdown
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, ElasticNet, SGDRegressor
from sklearn.decomposition import PCA

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
    output = os.path.join(path_to_data,"IT_data.h5")
    if not os.path.exists(output):
        url = "https://drive.google.com/file/d/1s6caFNRpyR9m7ZM6XEv_e8mcXT3_PnHS/view?usp=share_link"
        gdown.download(url, output, quiet=False, fuzzy=True)
    else:
        print("File already exists. Skipping download.")

def print_data_info(stimulus_train, spikes_train):
    n_stimulus, n_channels, img_size, _ = stimulus_train.shape
    _, n_neurons = spikes_train.shape
    print('The train dataset contains {} stimuli and {} IT neurons'.format(n_stimulus,n_neurons))
    print('Each stimulus have {} channels (RGB)'.format(n_channels))
    print('The size of the image is {}x{}'.format(img_size,img_size))

def select_model(model_str: str):
    models = {
        "linear_regression_cf": LinearRegression(),
        "ridge_cf": RidgeCV(),
        "lasso": Lasso(),
        "elastic_net": ElasticNet(),
        "sgd_linear": SGDRegressor(l2=100),
        "sgd_ridge": SGDRegressor(l1=100),
        "sgd_elastic_net": SGDRegressor(),
    }
    
    return models.get(model_str.lower(), None)

def compute_pca(X_train, X_val, n_components=20):
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
    return X_train, X_val 
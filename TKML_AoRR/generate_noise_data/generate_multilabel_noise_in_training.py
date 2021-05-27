import numpy as np
from sklearn import metrics
import scipy.io
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat, savemat
import os


def get_noise_label(label, noise_ratio=0, seed =0):
    random_state = np.random.RandomState(seed)
    noise_label = label.copy()
    n_labels = label.shape[1]
    n_samples = label.shape[0]
    if noise_ratio > 0:
        n_sample_noisy = int(noise_ratio * n_samples / 100)
        noise_sample_index = random_state.choice(np.asarray(range(n_samples)), n_sample_noisy, replace=False)
        for i in noise_sample_index:
            temp = np.zeros((1,n_labels),dtype=int)
            pos_labels_num = sum(label[i] == 1)
            noisy_label_index = random_state.choice(np.asarray(range(n_labels)), pos_labels_num, replace=False)
            temp[0,noisy_label_index]=1
            noise_label[i] = temp
    return noise_label

def generate_noise_label(dataname, data_path, seed, train_size, normal_type, noise_ratio):
    # dataname = "yeast" #'yeast', emotions scene
    if dataname == "yeast":
        feature_size = 103
    elif dataname == "emotions":
        feature_size = 72
    else: # scene
        feature_size = 294
    dataset = pd.read_csv(data_path, header=None, delimiter=',')
    ori_data = dataset.values[:, 0:feature_size].astype(float)
    if normal_type==0:
        data = -1 + (1 + 1) * (
                    (ori_data - np.min(ori_data, axis=0)) / (np.max(ori_data, axis=0) - np.min(ori_data, axis=0) + 1e-8))
    else:
        data = (ori_data - np.min(ori_data, axis=0)) / (np.max(ori_data, axis=0) - np.min(ori_data, axis=0) + 1e-8)
    label = dataset.values[:, feature_size:].astype(np.int)
    random_state = np.random.RandomState(seed)
    # # uniqueValues, occurCount = np.unique(label, return_counts=True)
    remaining_indices = list(range(len(label)))
    train_indices = random_state.choice(remaining_indices, int(len(label) * train_size), replace=False)
    test_indices = np.setdiff1d(remaining_indices, train_indices)
    X_train = data[train_indices, :]
    y_train = label[train_indices]

    X_test = data[test_indices, :]
    y_test = label[test_indices]

    y_train_noise = get_noise_label(y_train, noise_ratio, seed =seed)


    return X_train, y_train_noise, X_test, y_test
# from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import torch
import scipy.io
from PIL import Image
from TKML_AoRR.generate_noise_data.generate_multilabel_noise_in_training import generate_noise_label

class yeastDataset(Dataset):

    def __init__(self,x,y, transform=None, target_transform=None):
        data = []
        # x = x.transpose((0, 2, 3, 1))
        for i in range(len(x)):
            data.append((x[i],y[i]))
        self.data = data
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, index):

        feature, label = self.data[index]
        # feature = Image.fromarray(feature.astype(np.uint8))
        if self.transform is not None:
            feature = self.transform(feature)
        return feature, label

    def __len__(self):
        return len(self.data)

def yeast_dataloader_generation(dataname, data_path, seed, train_size, normal_type, noise_ratio):
    X_train, y_train_noise, X_test, y_test=\
        generate_noise_label(dataname, data_path, seed, train_size, normal_type, noise_ratio)
    X_train = np.float32(X_train)
    y_train = np.int64(y_train_noise)
    X_test = np.float32(X_test)
    y_test = np.int64(y_test)
    trainset = yeastDataset(X_train, y_train)
    testset = yeastDataset(X_test, y_test)
    return trainset, testset

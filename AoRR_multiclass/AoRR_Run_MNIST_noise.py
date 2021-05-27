import matplotlib.pyplot as plt
import matplotlib
import time
import scipy.io
import numpy as np
from sklearn import datasets, svm, metrics
# from keras.datasets import mnist
import numba
import gzip
import itertools
# from keras.utils import np_utils
from scipy.io import loadmat, savemat
from sklearn.preprocessing import OneHotEncoder
from logisticregression_Avg import LogisticRegression_Avg
from logisticregression_Avg_train_val import LogisticRegression_Avg_train_val
from logisticregression_ATK import LogisticRegression_ATK
from logisticregression_ATK_train_val import LogisticRegression_ATK_train_val
from logisticregression_Top1 import LogisticRegression_Top1
from logisticregression_Top1_train_val import LogisticRegression_Top1_train_val
from logisticregression_AoRR_GD import LogisticRegression_AoRR_GD
from logisticregression_AoRR_GD_train_val import LogisticRegression_AoRR_GD_train_val
from logisticregression_AoRR_GD_train import LogisticRegression_AoRR_GD_train
from logisticregression_AoRR_DC import LogisticRegression_AoRR_DC
from logisticregression_AoRR_DC_train_val import LogisticRegression_AoRR_DC_train_val

if __name__ == '__main__':

    ####### Parameters
    Model_name = "LogisticRegression"
    Data_name = 'MNIST'  ###MNIST

    # Method = "Avg"
    # Method = "ATK"
    # Method = "Top1"
    # Method = "AoRR_GD"
    # Method = "AoRR_DC"
    # Method = "AoRR_GD_train_val"
    Method = "AoRR_DC_train_val"
    # Method = "ATK_train_val"

    kvalue = 10000
    mvalue = 5000
    noise_ratio = 40
    data_name='MNIST_noise_{}'.format(noise_ratio)


    #######  Read data
    # mnist = input_data.read_data_sets("data/MNIST", one_hot=True)

    # mnist_noise = loadmat('datasets/{}_noise_data/mnist_noise_{}.mat'.format(Data_name, noise_ratio))
    # X_train = mnist_noise['X_train']
    # y_train_preprocess = mnist_noise['Y_train'][0]
    # onehot_encoder = OneHotEncoder(sparse=False)
    # y_train = y_train_preprocess.reshape(len(y_train_preprocess), 1)
    # y_train = onehot_encoder.fit_transform(y_train).astype(int)
    #
    # X_test_old = mnist_noise['X_test']
    # y_test_preprocess = mnist_noise['Y_test'][0]
    #
    # X_val = X_test_old[0:X_test_old.shape[0]//2]
    # X_test = X_test_old[X_test_old.shape[0]//2:]
    # y_val = y_test_preprocess[0:X_test_old.shape[0]//2].reshape(len(y_test_preprocess)//2, 1)
    # y_val = onehot_encoder.fit_transform(y_val).astype(int)
    # y_test = y_test_preprocess[X_test_old.shape[0]//2:].reshape(len(y_test_preprocess)//2, 1)
    # y_test = onehot_encoder.fit_transform(y_test).astype(int)

    onehot_encoder = OneHotEncoder(sparse=False)
    mnist_noise = loadmat('datasets/{}_noise_data/mnist_sym_noise_{}.mat'.format(Data_name, noise_ratio))
    X_train = mnist_noise['X_train']
    X_val = mnist_noise['X_val']
    X_test = mnist_noise['X_test']
    y_train = mnist_noise['Y_train'][0]
    y_train = onehot_encoder.fit_transform(y_train.reshape(len(y_train), 1)).astype(int)
    y_val = mnist_noise['Y_val'][0]
    y_val = onehot_encoder.fit_transform(y_val.reshape(len(y_val), 1)).astype(int)
    y_test = mnist_noise['Y_test'][0]
    y_test = onehot_encoder.fit_transform(y_test.reshape(len(y_test), 1)).astype(int)


    ####### Random show an image
    # example_idx = np.random.randint(len(X_train))
    # # example_idx = 4221
    # plt.imshow(np.reshape(X_train[example_idx], (28, 28)), cmap='gray')
    # plt.show()
    # print('Index: {}; Label: {}'.format(example_idx, y_train[example_idx]))

    #### Select classifier and set parameters
    if Method == "Avg":
        classifier = LogisticRegression_Avg(lr=0.1, reg=1e-5, num_iter=1000, seed=1234,
                                            ratio=noise_ratio)  #### average case
    elif Method == "ATK":
        classifier = LogisticRegression_ATK(lr=0.1, reg=1e-5, num_iter=5000, k_value=kvalue, seed=1234)  #### ATK case
    elif Method == "Top1":
        classifier = LogisticRegression_Top1(lr=0.01, reg=1e-5, num_iter=10000, seed=1234)  #### Top1 case
    elif Method == "AoRR_GD":
        classifier = LogisticRegression_AoRR_GD(lr=0.03, reg=1e-5, num_iter=5000, k_value=kvalue, m_value=mvalue,
                                                seed=1234)
    elif Method == "AoRR_DC":
        classifier = LogisticRegression_AoRR_DC(lr=0.2, reg=1e-5, outer_iter=100, inner_iter=1000, k_value=kvalue,
                                                m_value=mvalue, \
                                                dataname=data_name, Modelname=Model_name, seed=1234)
    elif Method == "AoRR_GD_train_val":
        classifier = LogisticRegression_AoRR_GD_train_val(lr_train=0.4, lr_val=0.5, reg=1e-5, num_iter=20000,
                                                          seed=1234, \
                                                          milestone=50, n_epochs_stop=50)
    elif Method == "Avg_GD_train_val":
        classifier = LogisticRegression_Avg_train_val(lr=0.1, reg=1e-5, num_iter=10000, seed=1234)
    elif Method == "AoRR_DC_train_val":
        classifier = LogisticRegression_AoRR_DC_train_val(lr_train=0.4, lr_val=0.5, train_iter=20000, reg=1e-5,
                                                          outer_iter=20, inner_iter=5000, \
                                                          dataname=data_name, Modelname=Model_name, seed=5678, milestone=50, n_epochs_stop=50)
    elif Method == "ATK_train_val":
        classifier = LogisticRegression_ATK_train_val(lr_train=0.4, lr_val=0.5, reg=1e-5, num_iter=20000, seed=1234,\
                                                      milestone=50, n_epochs_stop=50)

    #### Use classifier to fit data
    t_start = time.time()
    parameters = classifier.fit(X_train, y_train, X_val, y_val, X_test, y_test)
    t_total = time.time() - t_start
    print('Top-{} Accuracy (test) :'.format(1), classifier.predict_topk_acc(y_test, X_test, kk_value=1, X_modify = True))
    print('Top-{} Accuracy (test) :'.format(2), classifier.predict_topk_acc(y_test, X_test, kk_value=2, X_modify = True))
    print('Top-{} Accuracy (test) :'.format(3), classifier.predict_topk_acc(y_test, X_test, kk_value=3, X_modify = True))
    print('Top-{} Accuracy (test) :'.format(4), classifier.predict_topk_acc(y_test, X_test, kk_value=4, X_modify = True))
    print('Top-{} Accuracy (test) :'.format(5), classifier.predict_topk_acc(y_test, X_test, kk_value=5, X_modify = True))
    print("total_time: ", t_total)
    # np.save('parameters/MNIST_noise/'+ Model_name+'_'+Method+'_'+'noise_ratio'+'_'+str(noise_ratio)+ '.npy', parameters)

    print("shu")

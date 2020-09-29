import numpy as np
import pandas as pd
from TKML.TKML_DC import MultilabelSVM_TopK_DC_psi5_same_k

def main():
    print('Loading data...')
    dataname = "emotions"
    train_size = 0.50
    val_size = 0.25
    test_size = 1-train_size-val_size
    seed = 1

    dataset = pd.read_csv('./data/{}.dat'.format(dataname), header=None, delimiter=',')
    ori_data = dataset.values[:, 0:72].astype(float)
    #linear scale each dimension to [-1,1] or [0,1]
    data = -1 + (1 + 1) * ((ori_data - np.min(ori_data, axis=0)) / (np.max(ori_data, axis=0) - np.min(ori_data, axis=0) + 1e-8))

    label = dataset.values[:, 72:].astype(np.int)
    k_value_all = np.sum(label,axis=1)

    random_state = np.random.RandomState(seed)
    rp = random_state.permutation(data.shape[0])
    # rp = np.random.permutation(data.shape[0])
    data = data[rp,:]
    label = label[rp,:]
    Ntrain = np.ceil(data.shape[0] * train_size).astype(int)
    Nval = np.ceil(data.shape[0] * val_size).astype(int)

    X_train = data[0:Ntrain, :]
    y_train = label[0:Ntrain, :]

    X_val = data[Ntrain:(Ntrain + Nval), :]
    y_val = label[Ntrain:(Ntrain + Nval), :]

    X_test = data[(Ntrain + Nval):data.shape[0], :]
    y_test = label[(Ntrain + Nval):data.shape[0], :]

    print('Training multi-label SVM...')
    self_cs = MultilabelSVM_TopK_DC_psi5_same_k(K_value=5, number_iter=20, inner_iter =1000, learning_rate = 1e-1, dataseed=seed, seed = 1234, reg= 1e-4, dataname = dataname)

    self_cs.fit(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == '__main__':
    main()

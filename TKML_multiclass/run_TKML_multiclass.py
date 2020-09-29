import numpy as np
from sklearn import metrics
import scipy.io

from TKML_multiclass.TKML_multiclass_DC import MulticlassSVM_TopK_DC_psi5

def main():
    print('Loading data...')
    dataname = "mnist_noise_20"

    Toydata_path = 'data/{}.mat'.format(dataname)

    Toydata = scipy.io.loadmat(Toydata_path)

    X_train = Toydata['X_train'].astype(float)
    y_train = Toydata['Y_train'].flatten().astype(np.int)

    X_test = Toydata['X_test'].astype(float)
    y_test = Toydata['Y_test'].flatten().astype(np.int)

    print('Training self Crammer-Singer...')
    self_cs = MulticlassSVM_TopK_DC_psi5('crammer-singer', K_value=2, number_iter=21, inner_iter =2000, learning_rate = 1e-1, seed = 1234, dataname = dataname)

    self_cs.fit(X_train, y_train, X_test, y_test)
    print('Self Crammer-Singer Accuracy (train):',
          metrics.accuracy_score(y_train, self_cs.predict(X_train)))
    print('Self Crammer-Singer Accuracy (test) :',
          metrics.accuracy_score(y_test, self_cs.predict(X_test)))

    print('Top-{} Accuracy (test) :'.format(1), self_cs.predict_topk_acc(y_test, X_test, kk_value=1))
    print('Top-{} Accuracy (test) :'.format(2), self_cs.predict_topk_acc(y_test, X_test, kk_value=2))
    print('Top-{} Accuracy (test) :'.format(3), self_cs.predict_topk_acc(y_test, X_test, kk_value=3))
    print('Top-{} Accuracy (test) :'.format(4), self_cs.predict_topk_acc(y_test, X_test, kk_value=4))
    print('Top-{} Accuracy (test) :'.format(5), self_cs.predict_topk_acc(y_test, X_test, kk_value=5))
    # print('Top-{} Accuracy (test) :'.format(10), self_cs.predict_topk_acc(y_test, X_test, kk_value=10))


if __name__ == '__main__':
    main()

import numpy as np
import scipy.sparse as sp
import time
from scipy.special import expit
from scipy.special import digamma
from sklearn import metrics


class LogisticRegression_AoRR_DC_train_val:
    def __init__(self, lr_train=0.01, lr_val=0.01, train_iter =100, reg= 1e-5, outer_iter=100, inner_iter=1000, fit_intercept=True, \
                 verbose=True, seed =1234, dataname='real', Modelname='LogisticRegression', milestone = 50, n_epochs_stop=50):
        self.lr_train = lr_train
        self.lr_val = lr_val
        self.train_iter = train_iter
        self.reg = reg
        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.seed = seed
        self.milestone = milestone
        self.n_epochs_stop = n_epochs_stop
        self.dataname = dataname
        self.Model_name = Modelname

    def __add_intercept(self, X):
        return np.hstack([X, np.ones((len(X), 1))])

    def __softmax(self, x):
        x_shifted = x - np.max(x, axis=1).reshape((len(np.max(x, axis=1)), 1))
        sum = np.sum(np.exp(x_shifted), axis=1).reshape((len(np.sum(np.exp(x_shifted), axis=1)), 1))
        return np.exp(x_shifted) / sum

    def __softmax_ori(self, x):
        sum = np.sum(np.exp(x), axis=1).reshape((len(np.sum(np.exp(x), axis=1)), 1))
        return np.exp(x) / sum

    def __gradient(self, W, X, y):
        y_pred = np.matmul(W, X).T
        softmax_value = self.__softmax(y_pred)
        gradient = np.matmul(X, (softmax_value - y)).T
        return softmax_value, gradient

    def __score_and_loss_train(self, softmax_value, y):
        results = np.argmax(softmax_value, axis=1) == np.argmax(y, axis=1)
        loss = -np.log(softmax_value[y.astype(bool)])
        return np.mean(results), loss

    def __score_and_loss_test(self, W, X, y):
        y_pred = np.matmul(W, X).T
        y_pred = y_pred - np.max(y_pred, axis=1).reshape((len(np.max(y_pred, axis=1)), 1))
        sum = np.sum(np.exp(y_pred), axis=1).reshape((len(np.sum(np.exp(y_pred), axis=1)), 1))
        softmax_value = np.exp(y_pred) / sum
        results = np.argmax(softmax_value, axis=1) == np.argmax(y, axis=1)
        loss = -np.log(softmax_value[y.astype(bool)])
        return np.mean(results), loss

    def fit(self, X, y, X_val, y_val, X_test, y_test):
        print('X_train and y_train shape:', X.shape, y.shape, 'X_test and y_test shape:', X_test.shape, y_test.shape)
        if self.fit_intercept:
            X = self.__add_intercept(X)
            X_val = self.__add_intercept(X_val)
            X_test = self.__add_intercept(X_test)

        # weights initialization
        np.random.seed(self.seed)
        self.W = np.random.rand(y.shape[1], X.shape[1])
        # self.W = np.zeros((10, X.shape[1]))

        loss_val_m = 0
        loss_val_k = 0
        flag = 0
        min_val_loss = np.Inf
        epochs_no_improve = 0
        for epoch in range(self.outer_iter):
            if flag==0:
                for epoch_train in range(self.train_iter):
                    softmax_value, gradient = self.__gradient(self.W, np.transpose(X), y)
                    W_gradient_all = gradient / X.shape[0]
                    self.W -= self.lr_train * (W_gradient_all + self.reg * self.W)
                    accuracy_train, loss_train = self.__score_and_loss_train(softmax_value, y)
                    accuracy_val, loss_val = self.__score_and_loss_test(self.W, np.transpose(X_val), y_val)
                    loss_val_m = np.mean(loss_val) + 1 * np.std(loss_val)
                    loss_val_k = np.mean(loss_val) - 2 * np.std(loss_val)
                    accuracy_test, loss_test = self.__score_and_loss_test(self.W, np.transpose(X_test), y_test)
                    # print('1:',np.mean(loss_train),'2', min_train_loss)
                    print("AoRR_use_train epoch:{}".format(epoch_train),
                          "loss_train:{}".format(f'{np.mean(loss_train):.4f}'),
                          "accuracy_train:{}".format(f'{accuracy_train:.4f}'), "|", \
                          "loss_val:{}".format(f'{np.mean(loss_val):.4f}'),
                          "accuracy_val:{}".format(f'{accuracy_val:.4f}'), "|", \
                          "loss_test:{}".format(f'{np.mean(loss_test):.4f}'),
                          "accuracy_test:{}".format(f'{accuracy_test:.4f}'))
                    if np.mean(loss_val) < min_val_loss:
                        epochs_no_improve = 0
                        min_val_loss = np.mean(loss_val)
                        # if epoch_train>=10:
                        #     flag = 1
                        #     break

                    else:
                        if epoch_train > self.milestone:
                            epochs_no_improve += 1
                            # print('epochs_no_improve: ',epochs_no_improve)
                            if epochs_no_improve == self.n_epochs_stop:
                                flag = 1
                                print('early_stop_initial_train')
                                break
            else:
                y_pred = np.matmul(self.W, np.transpose(X)).T
                softmax_value = self.__softmax(y_pred)
                u = (softmax_value - y)
                loss = -np.log(softmax_value[y.astype(bool)])
                # sorted_loss = np.sort(loss)[::-1]
                # lamb = sorted_loss[self.m_value - 1]
                lamb_m = loss_val_m
                hinge = loss - lamb_m
                loss[hinge < 0] = 0
                print('m value estimate: {}'.format(np.count_nonzero(loss)))
                u[loss == 0] = 0
                # subgradient = np.matmul(np.transpose(X), u).T/ self.m_value
                subgradient = np.matmul(np.transpose(X), u).T / X.shape[0]

                np.random.seed(self.seed)
                self.W = np.random.rand(y.shape[1], X.shape[1])
                acc_list = []
                acc_inital = 0
                min_val_loss_inner = np.Inf
                epochs_no_improve_inner = 0
                for j in range(self.inner_iter):
                    y_pred_inner = np.matmul(self.W, np.transpose(X)).T
                    softmax_value_inner = self.__softmax(y_pred_inner)
                    u_inner = (softmax_value_inner - y)
                    loss_inner = -np.log(softmax_value_inner[y.astype(bool)])
                    # sorted_loss_inner = np.sort(loss_inner)[::-1]
                    # lamb_inner = sorted_loss_inner[self.k_value - 1]
                    lamb_inner = loss_val_k
                    hinge_inner = loss_inner - lamb_inner
                    loss_inner[hinge_inner < 0] = 0
                    u_inner[loss_inner == 0] = 0
                    # gradient = np.matmul(np.transpose(X), u_inner).T/ self.k_value
                    W_gradient_all = np.matmul(np.transpose(X), u_inner).T / X.shape[0] - subgradient

                    self.W -= self.lr_val * (W_gradient_all + self.reg * self.W)

                    accuracy_train_inner, loss_train_inner = self.__score_and_loss_train(softmax_value_inner, y)
                    accuracy_val_inner, loss_val_inner = self.__score_and_loss_test(self.W, np.transpose(X_val), y_val)
                    loss_val_k = np.mean(loss_val_inner) - 2 * np.std(loss_val_inner)
                    accuracy_test_inner, loss_test_inner = self.__score_and_loss_test(self.W, np.transpose(X_test),
                                                                                      y_test)
                    print("inner_epoch:{}".format(j), "loss_train:{}".format(f'{np.mean(loss_train_inner):.4f}'),
                          "accuracy_train:{}".format(f'{accuracy_train_inner:.4f}'), "|", \
                          "loss_val:{}".format(f'{np.mean(loss_val_inner):.4f}'),
                          "accuracy_val:{}".format(f'{accuracy_val_inner:.4f}'), "|", \
                          "loss_test:{}".format(f'{np.mean(loss_test_inner):.4f}'),
                          "accuracy_test:{}".format(f'{accuracy_test_inner:.4f}'))
                    if accuracy_val_inner >= acc_inital:
                        np.save('./immedia_parameter/{}_{}_{}_W.npy'.format(self.dataname, self.Model_name, self.seed), self.W)
                    acc_list.append(accuracy_val_inner)
                    acc_inital = max(acc_list)


                    # if accuracy_val_inner> acc_inital:
                    #     epochs_no_improve_inner = 0
                    # else:
                    #     if j > self.milestone:
                    #         epochs_no_improve_inner += 1
                    #         # print('epochs_no_improve: ',epochs_no_improve)
                    #         if epochs_no_improve_inner == self.n_epochs_stop:
                    #             print('early_stop_inner_train')
                    #             break


                    # if np.mean(loss_val_inner) < min_val_loss_inner:
                    #     epochs_no_improve_inner = 0
                    #     min_val_loss_inner = np.mean(loss_val_inner)
                    # else:
                    #     if j > self.milestone:
                    #         epochs_no_improve_inner += 1
                    #         # print('epochs_no_improve: ',epochs_no_improve)
                    #         if epochs_no_improve_inner == self.n_epochs_stop:
                    #             print('early_stop_inner_train')
                    #             break

                self.W = np.load('./immedia_parameter/{}_{}_{}_W.npy'.format(self.dataname, self.Model_name, self.seed))
                accuracy_train_outer, loss_train_outer = self.__score_and_loss_train(softmax_value, y)
                accuracy_val_outer, loss_val_outer = self.__score_and_loss_test(self.W, np.transpose(X_val), y_val)
                loss_val_m = np.mean(loss_val_outer) + 1 * np.std(loss_val_outer)
                loss_val_k = np.mean(loss_val_outer) - 2 * np.std(loss_val_outer)
                accuracy_test_outer, loss_test_outer = self.__score_and_loss_test(self.W, np.transpose(X_test), y_test)
                print("outer_epoch:{}".format(epoch), "loss_train:{}".format(f'{np.mean(loss_train_outer):.4f}'),
                      "accuracy_train:{}".format(f'{accuracy_train_outer:.4f}'), "|", \
                      "loss_val:{}".format(f'{np.mean(loss_val_outer):.4f}'),
                      "accuracy_val:{}".format(f'{accuracy_val_outer:.4f}'), "|", \
                      "loss_test:{}".format(f'{np.mean(loss_test_outer):.4f}'),
                      "accuracy_test:{}".format(f'{accuracy_test_outer:.4f}'))
        return self.W

    def predict_topk_acc(self, Y, X, kk_value=5, X_modify = True):
        if X_modify == True:
            X_intercept = np.hstack([X, np.ones((len(X), 1))])
        else:
            X_intercept = X
        y_pred = np.matmul(self.W, np.transpose(X_intercept)).T
        softmax_value = self.__softmax(y_pred)
        top_k_acc_list = np.argsort(softmax_value, axis=1)
        count = 0
        for i in range(len(Y)):
            if np.argmax(Y, axis=1)[i] in top_k_acc_list[i][-kk_value:]:
                count = count + 1
        acc = count / len(Y)
        return acc


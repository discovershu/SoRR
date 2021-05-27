import numpy as np
import scipy.sparse as sp
import time
from scipy.special import expit
from scipy.special import digamma
from sklearn import metrics

class LogisticRegression_ATK_train_val:
    def __init__(self, lr_train=0.4, lr_val=0.2, reg= 1e-5, num_iter=10001, fit_intercept=True, verbose=True, seed =1234, milestone=50, n_epochs_stop=50):
        self.lr_train = lr_train
        self.lr_val = lr_val
        self.reg = reg
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.seed = seed
        self.milestone = milestone
        self.n_epochs_stop = n_epochs_stop

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

    def fit(self, X, y,X_val, y_val, X_test, y_test):
        print('X_train and y_train shape:', X.shape, y.shape, 'X_test and y_test shape:', X_test.shape, y_test.shape)
        if self.fit_intercept:
            X = self.__add_intercept(X)
            X_val = self.__add_intercept(X_val)
            X_test = self.__add_intercept(X_test)

        # weights initialization
        np.random.seed(self.seed)
        self.W = np.random.rand(y.shape[1],X.shape[1])
        # self.W = np.zeros((10, X.shape[1]))

        loss_val_m = 0
        loss_val_k = 0
        flag = 0
        min_val_loss = np.Inf
        epochs_no_improve = 0
        min_val_loss_atk = np.Inf
        epochs_no_improve_atk = 0

        for epoch in range(self.num_iter):
            if flag==0:
                softmax_value, gradient = self.__gradient(self.W, np.transpose(X), y)
                W_gradient_all = gradient / X.shape[0]
                self.W -= self.lr_train * (W_gradient_all + self.reg * self.W)
                accuracy_train, loss_train = self.__score_and_loss_train(softmax_value, y)
                accuracy_val, loss_val = self.__score_and_loss_test(self.W, np.transpose(X_val), y_val)
                loss_val_m = np.max(loss_val)
                loss_val_k = np.mean(loss_val) + 1 * np.std(loss_val)
                accuracy_test, loss_test = self.__score_and_loss_test(self.W, np.transpose(X_test), y_test)
                if np.mean(loss_val)<min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = np.mean(loss_val)
                    # if epoch>=200:
                    #     flag = 1
                else:
                    if epoch > self.milestone:
                        epochs_no_improve += 1
                        # print('epochs_no_improve: ',epochs_no_improve)
                        if epochs_no_improve == self.n_epochs_stop:
                            print('early_stop_initial_train')
                            flag = 1
                print("use_train epoch:{}".format(epoch), "loss_train:{}".format(f'{np.mean(loss_train):.4f}'),
                      "accuracy_train:{}".format(f'{accuracy_train:.4f}'), "|", \
                      "loss_val:{}".format(f'{np.mean(loss_val):.4f}'),
                      "accuracy_val:{}".format(f'{accuracy_val:.4f}'), "|", \
                      "loss_test:{}".format(f'{np.mean(loss_test):.4f}'),
                      "accuracy_test:{}".format(f'{accuracy_test:.4f}'))
            else:
                y_pred = np.matmul(self.W, np.transpose(X)).T
                softmax_value = self.__softmax(y_pred)
                u = (softmax_value - y)
                loss = -np.log(softmax_value[y.astype(bool)])
                # sorted_loss = np.sort(loss)[::-1]

                lamb = loss_val_k
                # lamb = 0
                hinge = loss - lamb
                loss[hinge < 0] = 0
                u[loss == 0] = 0
                gradient = np.matmul(np.transpose(X), u).T

                # W_gradient_all = gradient / self.k_value
                W_gradient_all = gradient / X.shape[0]
                # self.W -= self.lr_val * (W_gradient_all)
                self.W -= self.lr_val * (W_gradient_all + self.reg * self.W)

                accuracy_train, loss_train = self.__score_and_loss_train(softmax_value, y)
                accuracy_val, loss_val = self.__score_and_loss_test(self.W, np.transpose(X_val), y_val)
                # loss_val_m = np.max(loss_val)
                loss_val_k = np.mean(loss_val) - 2 * np.std(loss_val)
                accuracy_test, loss_test = self.__score_and_loss_test(self.W, np.transpose(X_test), y_test)

                print("epoch:{}".format(epoch), "loss_train:{}".format(f'{np.mean(loss_train):.4f}'),
                      "accuracy_train:{}".format(f'{accuracy_train:.4f}'), "|", \
                      "loss_val:{}".format(f'{np.mean(loss_val):.4f}'),
                      "accuracy_val:{}".format(f'{accuracy_val:.4f}'), "|", \
                      "loss_test:{}".format(f'{np.mean(loss_test):.4f}'),
                      "accuracy_test:{}".format(f'{accuracy_test:.4f}')
                      )
                if np.mean(loss_val) < min_val_loss_atk:
                    epochs_no_improve_atk = 0
                    min_val_loss_atk = np.mean(loss_val)
                else:
                    if epoch > self.milestone:
                        epochs_no_improve_atk += 1
                        # print('epochs_no_improve: ',epochs_no_improve)
                        if epochs_no_improve_atk == self.n_epochs_stop:
                            print('early_stop_validation_train')
                            break
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
                count = count +1
        acc = count / len(Y)
        return acc

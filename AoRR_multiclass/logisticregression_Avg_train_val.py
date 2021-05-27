import numpy as np
import scipy.sparse as sp
import time
from scipy.special import expit
from scipy.special import digamma
from sklearn import metrics
import math


class LogisticRegression_Avg_train_val:
    def __init__(self, lr=0.01, reg= 1e-5, num_iter=10001, fit_intercept=True, verbose=True, seed =1234, ratio = 0.1):
        self.lr = lr
        self.reg = reg
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.seed = seed
        self.ratio = ratio

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
        results = np.argmax(softmax_value, axis=1)==np.argmax(y, axis=1)
        loss = -np.log(softmax_value[y.astype(bool)])
        return np.mean(results), loss

    def __score_and_loss_test(self, W, X, y):
        y_pred = np.matmul(W, X).T
        y_pred = y_pred-np.max(y_pred, axis=1).reshape((len(np.max(y_pred, axis=1)), 1))
        sum = np.sum(np.exp(y_pred), axis=1).reshape((len(np.sum(np.exp(y_pred), axis=1)), 1))
        softmax_value = np.exp(y_pred) / sum
        results = np.argmax(softmax_value, axis=1)==np.argmax(y, axis=1)
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
        self.W = np.random.rand(y.shape[1],X.shape[1])
        # self.W = np.zeros((10, X.shape[1]))

        for epoch in range(self.num_iter):

            softmax_value, gradient = self.__gradient(self.W, np.transpose(X), y)
            W_gradient_all = gradient/X.shape[0]
            self.W -= self.lr * (W_gradient_all + self.reg * self.W)
            accuracy_train, loss_train = self.__score_and_loss_train(softmax_value, y)
            accuracy_val, loss_val = self.__score_and_loss_test(self.W, np.transpose(X_val), y_val)
            accuracy_test, loss_test = self.__score_and_loss_test(self.W, np.transpose(X_test), y_test)

            # if ((epoch+1)%200==0):
            #     np.save('result/avg/Synthetic/data1_noise/noise_{}_loss_epoch{}.npy'.format(self.ratio, epoch+1),loss_train)
            #     sorted_loss_index = np.argsort(loss_train)[::-1]
            #     np.save('result/avg/Synthetic/data1_noise/noise_{}_sorted_index_epoch{}.npy'.format(self.ratio,epoch+1),sorted_loss_index)
            #     sorted_loss_train = np.sort(loss_train)[::-1]
            #     loss_diff = sorted_loss_train[0:-1]-sorted_loss_train[1:]
            #     np.save('result/avg/Synthetic/data1_noise/noise_{}_sorted_loss_diff_epoch{}.npy'.format(self.ratio,epoch + 1), loss_diff)

            print("epoch:{}".format(epoch), "loss_train:{}".format(f'{np.mean(loss_train):.4f}'),
                  "accuracy_train:{}".format(f'{accuracy_train:.4f}'), "|", \
                  "loss_val:{}".format(f'{np.mean(loss_val):.4f}'),
                  "accuracy_val:{}".format(f'{accuracy_val:.4f}'), "|", \
                  "loss_test:{}".format(f'{np.mean(loss_test):.4f}'),
                  "accuracy_test:{}".format(f'{accuracy_test:.4f}'))
        return self.W
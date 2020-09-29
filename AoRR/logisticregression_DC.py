import numpy as np
from sklearn import metrics


class LogisticRegression_TOPK_DC:
    def __init__(self, lr=0.01, num_iter=100, inner_iter=1000, fit_intercept=True, verbose=True, k_value=5, k2_value = 4, seed =1234, dataname='real', Model_name='LogisticRegression'):
        self.lr = lr
        self.num_iter = num_iter
        self.inner_iter = inner_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.k_value = k_value
        self.k2_value = k2_value
        self.seed = seed
        self.dataname = dataname
        self.Model_name = Model_name

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, h, y):
        h = np.clip(h, 1e-7, 1.0 - 1e-7)  # clip h, incase h == 0
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def __individualloss(self, h, y):
        h = np.clip(h, 1e-7, 1.0 - 1e-7)  # clip h, incase h == 0
        return -y * np.log(h) - (1 - y) * np.log(1 - h)

    def fit(self, X, y,X_val,y_val,X_test, y_test):
        # X = X.todense()
        # X = np.asarray(X)
        np.random.seed(self.seed)
        print('X and y shape:', X.shape, y.shape)

        if self.fit_intercept:
            X = self.__add_intercept(X)
            X_val = self.__add_intercept(X_val)
            X_test = self.__add_intercept(X_test)

        # weights initialization
        self.theta = np.random.rand(X.shape[1])
        #        print('====',self.theta.shape)
        # acc_list = []
        # acc_inital = 0
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss_1 = self.__individualloss(h, y)
            sorted_loss = np.sort(loss_1)[::-1]
            lamb_1 = sorted_loss[self.k2_value-1]
            hinge_1 = loss_1 - lamb_1
            loss_1[hinge_1 < 0] = 0
            u = (h - y)
            u[loss_1==0]=0

            subgradient = np.dot(X.T, u) / y.size

            # np.random.seed(1234)
            # self.theta = np.random.rand(X.shape[1])
            self.theta = np.zeros((self.theta.shape[0],))
            acc_list = []
            acc_inital = 0
            for j in range(self.inner_iter):
                h_2 = self.__sigmoid(np.dot(X, self.theta))
                loss_2 = self.__individualloss(h_2, y)
                sorted_loss_2 = np.sort(loss_2)[::-1]
                lamb_2 = sorted_loss_2[self.k_value-1]
                hinge_2 = loss_2 - lamb_2
                loss_2[hinge_2 < 0] = 0

                v = (h_2 - y)
                v[loss_2==0]=0
                gradient = np.dot(X.T, v) / y.size - subgradient

                self.theta -= self.lr * gradient

                z_check = np.dot(X_val, self.theta)
                h_check = self.__sigmoid(z_check)
                accuracy = metrics.classification_report(y_val, h_check.round(), digits=3, output_dict=True)['accuracy']
                print("inner iter:",j,"acc: ", accuracy)

                if accuracy >= acc_inital:
                    np.save('./immedia_parameter/{}_{}_theta.npy'.format(self.dataname,self.Model_name), self.theta)
                acc_list.append(accuracy)
                acc_inital = max(acc_list)

            self.theta = np.load('./immedia_parameter/{}_{}_theta.npy'.format(self.dataname,self.Model_name))
            if (self.verbose == True and i % 1 == 0):
                z = np.dot(X, self.theta)
                z_val = np.dot(X_val, self.theta)
                z_test = np.dot(X_test, self.theta)
                h = self.__sigmoid(z)
                h_val = self.__sigmoid(z_val)
                h_test = self.__sigmoid(z_test)
                accuracy = metrics.classification_report(y, h.round(), digits=3 , output_dict=True)['accuracy']
                accuracy_val = metrics.classification_report(y_val, h_val.round(), digits=3, output_dict=True)[
                    'accuracy']
                accuracy_test = metrics.classification_report(y_test, h_test.round(), digits=3, output_dict=True)[
                    'accuracy']
                print("loss:", self.__loss(h, y), "accuracy:", accuracy, "accuracy_val:", accuracy_val,
                      "accuracy_test:", accuracy_test)
                # print("loss:", self.__loss(h, y), "accuracy:", accuracy)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sigmoid(np.dot(X, self.theta))

    def predict(self, X):
        # X = X.todense()
        # X = np.asarray(X)
        return self.predict_prob(X).round(), self.theta
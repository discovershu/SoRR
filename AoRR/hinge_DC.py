import numpy as np
from sklearn import metrics


class hinge_TOPK_DC:
    def __init__(self, lr=0.01, num_iter=100, inner_iter=1000, fit_intercept=True, verbose=True, k_value=5, k2_value = 4, early_stop=20, seed = 1234,dataname='real', Model_name='LogisticRegression'):
        self.lr = lr
        self.num_iter = num_iter
        self.inner_iter = inner_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.k_value = k_value
        self.k2_value = k2_value
        self.early_stop = early_stop
        self.seed = seed
        self.dataname = dataname
        self.Model_name = Model_name

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __sgn(self, z):
        z[z < 0] = -1
        z[z >= 0] = 1
        return z

    def __loss(self, h, y):
        # h = np.clip(h, 1e-7, 1.0 - 1e-7)  # clip h, incase h == 0

        loss = 1 - y * h
        loss[loss <= 0] = 0

        return loss.mean()

    def __individualloss(self, h, y):
        loss = 1 - y * h
        loss[loss <= 0] = 0

        return loss

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
        # acc_list = []
        # acc_inital = 0
        #        print('====',self.theta.shape)
        for i in range(self.num_iter):
            h = np.dot(X, self.theta)
            loss_1 = self.__individualloss(h, y)
            sorted_loss = np.sort(loss_1)[::-1]
            lamb_1 = sorted_loss[self.k2_value-1]
            hinge_1 = loss_1 - lamb_1
            loss_1[hinge_1 < 0] = 0
            sgn = np.ones((len(y), 1))
            sgn[loss_1 <= 0] = 0

            subgradient = -np.dot((sgn*X).T, y)/y.size

            self.theta = np.zeros((self.theta.shape[0],))
            val_loss_list = []
            acc_list = []
            acc_inital = 0

            for j in range(self.inner_iter):
                h_2 = np.dot(X, self.theta)
                loss_2 = self.__individualloss(h_2, y)
                sorted_loss_2 = np.sort(loss_2)[::-1]
                lamb_2 = sorted_loss_2[self.k_value-1]
                hinge_2 = loss_2 - lamb_2
                loss_2[hinge_2 < 0] = 0

                sgn = np.ones((len(y), 1))
                sgn[loss_2 <= 0] = 0
                gradient = -np.dot((sgn*X).T, y)/y.size - subgradient
                # gradient = -np.dot((sgn * X).T, y) / y.size

                self.theta -= self.lr * gradient

                z = np.dot(X_val, self.theta)
                val_loss = self.__loss(z, y_val)
                # print(val_loss)
                z = self.__sgn(z)
                # val_mis_acc = self.__loss(z, y)
                accuracy = metrics.classification_report(y_val, z, digits=3 , output_dict=True)['accuracy']
                if accuracy >= acc_inital:
                    np.save('./immedia_parameter/{}_{}_theta.npy'.format(self.dataname,self.Model_name), self.theta)
                acc_list.append(accuracy)
                acc_inital = max(acc_list)

                # print(val_mis_acc, accuracy)
                val_loss_list.append(val_loss)
                print("inner:",j, "loss:", val_loss, "accuracy:", accuracy)

            self.theta = np.load('./immedia_parameter/{}_{}_theta.npy'.format(self.dataname,self.Model_name))
            if (self.verbose == True and i % 1 == 0):
                outer = np.dot(X, self.theta)
                outer_val = np.dot(X_val, self.theta)
                outer_test = np.dot(X_test, self.theta)
                val_loss_outer = self.__loss(outer, y)
                outer = self.__sgn(outer)
                outer_val = self.__sgn(outer_val)
                outer_test = self.__sgn(outer_test)
                accuracy2 = metrics.classification_report(y, outer, digits=3, output_dict=True)['accuracy']
                accuracy2_val = metrics.classification_report(y_val, outer_val, digits=3, output_dict=True)[
                    'accuracy']
                accuracy2_test = metrics.classification_report(y_test, outer_test, digits=3, output_dict=True)[
                    'accuracy']
                print("loss:", self.__loss(h, y), "accuracy:", accuracy2, "accuracy_val:", accuracy2_val,
                      "accuracy_test:", accuracy2_test)
                # print("outer:", "loss:", val_loss_outer, "accuracy:", accuracy2)

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        return self.__sgn(np.dot(X, self.theta))

    def predict(self, X):
        # X = X.todense()
        # X = np.asarray(X)
        return self.predict_prob(X), self.theta
import numpy as np
from sklearn import metrics


class MultilabelSVM_TopK_DC_psi5_same_k:

    def __init__(self, K_value=2, number_iter=10, inner_iter =150, learning_rate = 1e-2, dataseed=1, seed = 1234, reg=1e-4, dataname = 'data'):
        self.labels = None
        self.binary_svm = None
        self.W = None
        self.K_value = K_value
        self.number_iter = number_iter
        self.inner_iter = inner_iter
        self.learning_rate = learning_rate
        self.seed = seed
        self.dataname = dataname
        self.dataseed = dataseed
        self.reg = reg

    def fit(self, X, y, X_val, y_val, X_test, y_test):
        self.fit_cs(X, y, X_val, y_val, X_test, y_test)

    def fit_cs(self, X, y, X_val, y_val, X_test, y_test):
        print("Model: DC_psi5", 'k value', self.K_value,'outer number_iter:', self.number_iter, 'inner number_iter:',
              self.inner_iter, 'learning_rate:',self.learning_rate, 'dataseed', self.dataseed, 'seed', self.seed)
        # self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        X_test_intercept = np.hstack([X_test, np.ones((len(X_test), 1))])
        X_val_intercept = np.hstack([X_val, np.ones((len(X_val), 1))])

        N, d = X_intercept.shape
        K = y.shape[1]

        GT_label_index = []
        for i in range(y.shape[0]):
            GT_label_index.append(np.where(y[i] == 1)[0])

        GT_label_index_val = []
        for i in range(y_val.shape[0]):
            GT_label_index_val.append(np.where(y_val[i] == 1)[0])

        GT_label_index_test = []
        for i in range(y_test.shape[0]):
            GT_label_index_test.append(np.where(y_test[i] == 1)[0])

        # W = np.zeros((K, d))
        np.random.seed(self.seed)
        self.W = np.random.rand(K, d)

        # out_iter = 10
        # n_iter = 150
        # learning_rate = 1e-2
        test_predict = []
        for i in range(self.number_iter):
            loss_outer, grad_outer =self.subgrad_student(self.W, X_intercept, y, GT_label_index)
            subgradient = grad_outer
            np.random.seed(self.seed)
            self.W =  np.random.rand(K, d)
            # W = np.zeros((K, d))
            acc_list = []
            acc_initial = 0
            for j in range(self.inner_iter):
                loss_inner, grad_inner = self.grad_student(self.W, X_intercept, y, subgradient, GT_label_index)
                self.W -= self.learning_rate * (grad_inner / N + self.reg * self.W)
                # self.W -= self.learning_rate * (grad_inner/N)

                ##train

                predict_values_train = self.predict(X)
                top_k_predict_labels_train = self.predict_top_k_labels(predict_values_train)
                acc_train = self.topk_acc_metric(top_k_predict_labels_train, GT_label_index)
                ##val

                predict_values_val = self.predict(X_val)
                top_k_predict_labels_val = self.predict_top_k_labels(predict_values_val)
                acc_val = self.topk_acc_metric(top_k_predict_labels_val, GT_label_index_val)
                ##test

                predict_values_test = self.predict(X_test)
                top_k_predict_labels_test = self.predict_top_k_labels(predict_values_test)
                acc_test = self.topk_acc_metric(top_k_predict_labels_test, GT_label_index_test)

                if acc_val >= acc_initial:
                    np.save('./immedia_parameter/{}_weight_{}_{}_{}_{}_{}_{}_{}.npy'.format(self.dataname,self.number_iter, self.inner_iter, self.learning_rate, self.seed, self.dataseed,self.reg, self.K_value), self.W)

                acc_list.append(acc_val)
                acc_initial = max(acc_list)

                print("inner_epoch:", j, 'tr_l={:.3f} tr_a={:.5f}:'.format(loss_inner,acc_train),'|'\
                      ,'va_l={:.3f} va_a={:.5f}:'.format(self.loss_student(self.W, X_val_intercept, y_val, GT_label_index_val), acc_val),'|'\
                      ,'te_l={:.3f} te_a={:.5f}:'.format(self.loss_student(self.W, X_test_intercept, y_test,GT_label_index_test),acc_test))
            self.W = np.load('./immedia_parameter/{}_weight_{}_{}_{}_{}_{}_{}_{}.npy'.format(self.dataname,self.number_iter, self.inner_iter, self.learning_rate, self.seed,self.dataseed,self.reg, self.K_value))
            # self.W = W
            # print(W)

            ####train
            predict_values_train_outer = self.predict(X)
            top_k_predict_labels_train_outer = self.predict_top_k_labels(predict_values_train_outer)
            acc_train_outer = self.topk_acc_metric(top_k_predict_labels_train_outer, GT_label_index)

            ####val
            predict_values_val_outer = self.predict(X_val)
            top_k_predict_labels_val_outer = self.predict_top_k_labels(predict_values_val_outer)
            acc_val_outer = self.topk_acc_metric(top_k_predict_labels_val_outer, GT_label_index_val)

            ####test

            predict_values_test_outer = self.predict(X_test)
            top_k_predict_labels_test_outer = self.predict_top_k_labels(predict_values_test_outer)
            acc_test_outer = self.topk_acc_metric(top_k_predict_labels_test_outer, GT_label_index_test)

            print("epoch:", i, 'tr_l={:.3f} tr_a={:.5f}:'.format(loss_outer, acc_train_outer), '|'\
                  ,'va_l={:.3f} va_a={:.5f}:'.format(self.loss_student(self.W, X_val_intercept, y_val, GT_label_index_val), acc_val_outer),'|'\
                  ,'te_l={:.3f} te_a={:.5f}:'.format(self.loss_student(self.W, X_test_intercept, y_test,GT_label_index_test),acc_test_outer))
        # self.W = W
        # np.save('./predict/{}_predict_{}_{}_{}_{}_{}.npy'.format(self.dataname, self.K_value, self.number_iter,
        #                                                                   self.inner_iter, self.learning_rate,
        #                                                                   self.seed), test_predict)

    def predict(self, X):
        return self.predict_cs(X)


    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return X_intercept.dot((self.W).T)

    def predict_labels(self, predict_values, GT_label_index):
        labels = []
        for i in range(predict_values.shape[0]):
            a = predict_values[i].argsort()[-len(GT_label_index[i]):][::-1]
            b = np.zeros(predict_values.shape[1],int)
            b[a]=1
            labels.append(np.asarray(b))
        return np.asarray(labels)

    def predict_top_k_labels(self, predict_values):
        labels = []
        for i in range(predict_values.shape[0]):
            a = predict_values[i].argsort()[-self.K_value:][::-1]
            labels.append(np.asarray(a))
        return np.asarray(labels)

    def HM_metric(self, true_labels, pred_labels):
        count = 0
        for i in range(true_labels.shape[0]):
            count = count + metrics.hamming_loss(true_labels[i],pred_labels[i])
        return count/true_labels.shape[0]

    def recall_metric(self, true_labels, pred_labels):
        count = 0
        for i in range(true_labels.shape[0]):
            count = count + metrics.recall_score(true_labels[i], pred_labels[i])
        return count / true_labels.shape[0]

    def AP_metric(self, true_labels, pred_labels):
        count = 0
        for i in range(true_labels.shape[0]):
            count = count + metrics.average_precision_score(true_labels[i], pred_labels[i])
        return count / true_labels.shape[0]

    def topk_acc_metric(self, top_k_predict_labels_list, GT_label_index_list):
        count = 0
        for i in range(top_k_predict_labels_list.shape[0]):
            if self.K_value>GT_label_index_list[i].shape[0]:
                count = count + int(set(top_k_predict_labels_list[i]).issuperset(GT_label_index_list[i]))
            else:
                count = count + int(set(GT_label_index_list[i]).issuperset(top_k_predict_labels_list[i]))
        return count/top_k_predict_labels_list.shape[0]


    def loss_student(self, W, X, y,GT_label_index, C=1.0):
        """
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        """
        if self.labels is None:
            self.labels = np.unique(y)

        # loss of regularization term
        l2_loss = 0.5 * np.sum(W**2)

        # gradient of the other term
        # get the matrix of term 1 - delta(j, y_i) + w_j^T * x_i
        scores = np.matmul(W, np.transpose(X))
        argmin_GT = []
        for i in range(y.shape[0]):
            GT_score = scores[GT_label_index[i], i]
            argmin_GT.append(GT_label_index[i][np.argmin(GT_score)])
        loss_modify = 1 + scores - scores[np.array(argmin_GT), np.arange(X.shape[0])].reshape(1, X.shape[0])
        loss_modify = (loss_modify > 0) * loss_modify

        # total_loss = l2_loss + multiclass_loss
        total_loss = loss_modify.sum() / X.shape[0]
        return total_loss

    def grad_student(self, W, X, y, sub_gradient, GT_label_index, C=1.0):
        """
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The gradient of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        """
        if self.labels is None:
            self.labels = np.unique(y)

        # gradient of regularization term
        l2_grad = W

        scores = np.matmul(W, np.transpose(X))
        argmin_GT = []
        for i in range(y.shape[0]):
            GT_score = scores[GT_label_index[i], i]
            argmin_GT.append(GT_label_index[i][np.argmin(GT_score)])

        loss_modify = 1 + scores - scores[np.array(argmin_GT), np.arange(X.shape[0])].reshape(1, X.shape[0])
        loss_modify = (loss_modify > 0) * loss_modify
        sorted_loss_modify = np.sort(loss_modify, axis=0)[::-1]
        lamb = np.maximum(0, sorted_loss_modify[self.K_value, np.arange(X.shape[0])])
        hinge = loss_modify - lamb
        loss_modify[hinge < 0] = 0
        loss = loss_modify.sum()/X.shape[0]

        ##calculate gradient
        loss_modify_index = (loss_modify > 0) * 1
        column_sum = np.sum(loss_modify_index, axis=0)
        loss_modify_index[np.array(argmin_GT), np.arange(X.shape[0])] = -column_sum
        total_grad = loss_modify_index.dot(X) - sub_gradient

        return loss, total_grad

    def subgrad_student(self, W, X, y, GT_label_index, C=1.0):
        """
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arguments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The gradient of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        """
        if self.labels is None:
            self.labels = np.unique(y)

        # gradient of regularization term
        l2_grad = W

        scores = np.matmul(W, np.transpose(X))
        argmin_GT=[]
        for i in range(y.shape[0]):
            GT_score=scores[GT_label_index[i],i]
            argmin_GT.append(GT_label_index[i][np.argmin(GT_score)])

        loss_modify = 1 + scores - scores[np.array(argmin_GT), np.arange(X.shape[0])].reshape(1, X.shape[0])
        loss_modify = (loss_modify > 0) * loss_modify
        sorted_loss_modify = np.sort(loss_modify, axis=0)[::-1]
        lamb = np.maximum(0, sorted_loss_modify[self.K_value-1, np.arange(X.shape[0])])
        hinge = loss_modify - lamb
        loss_modify[hinge < 0] = 0
        loss = loss_modify.sum() / X.shape[0]

        ##calculate gradient
        loss_modify_index = (loss_modify>0)*1
        column_sum = np.sum(loss_modify_index,axis=0)
        loss_modify_index[np.array(argmin_GT), np.arange(X.shape[0])]=-column_sum
        total_grad = loss_modify_index.dot(X)


        return loss, total_grad


import numpy as np
from sklearn import metrics


class MulticlassSVM_TopK_DC_psi5:

    def __init__(self, mode, K_value=2, number_iter=10, inner_iter =150, learning_rate = 1e-2, seed = 1234, dataname = 'data'):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode
        self.labels = None
        self.binary_svm = None
        self.W = None
        self.K_value = K_value
        self.number_iter = number_iter
        self.inner_iter = inner_iter
        self.learning_rate = learning_rate
        self.seed = seed
        self.dataname = dataname

    def fit(self, X, y, X_test, y_test):
        if self.mode == 'crammer-singer':
            self.fit_cs(X, y, X_test, y_test)

    def fit_cs(self, X, y, X_test, y_test):
        print("Model: DC_psi5", 'K_value:', self.K_value, 'outer number_iter:', self.number_iter, 'inner number_iter:',
              self.inner_iter, 'learning_rate:',
              self.learning_rate)
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        X_test_intercept = np.hstack([X_test, np.ones((len(X_test), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        # W = np.zeros((K, d))
        np.random.seed(self.seed)
        self.W = np.random.rand(K, d)

        # out_iter = 10
        # n_iter = 150
        # learning_rate = 1e-2
        test_predict = []
        for i in range(self.number_iter):
            subgradient = self.subgrad_student(self.W, X_intercept, y)
            np.random.seed(self.seed)
            self.W =  np.random.rand(K, d)
            # W = np.zeros((K, d))
            acc_list = []
            acc_initial = 0
            for j in range(self.inner_iter):
                self.W -= self.learning_rate * (self.grad_student(self.W, X_intercept, y, subgradient)/N)

                acc_test = metrics.accuracy_score(y_test, self.predict(X_test))
                if acc_test >= acc_initial:
                    np.save('./immedia_parameter/{}_weight_{}_{}_{}_{}_{}.npy'.format(self.dataname,self.K_value, self.number_iter, self.inner_iter, self.learning_rate, self.seed), self.W)

                acc_list.append(acc_test)
                acc_initial = max(acc_list)

                print("inner_epoch:", j, 'tr_l={:.3f} tr_a={:.5f}:'.format(self.loss_student(self.W, X_intercept, y),
                                                                           metrics.accuracy_score(y, self.predict(X))),
                      'te_l={:.3f} te_a={:.5f}:'.format(self.loss_student(self.W, X_test_intercept, y_test),
                                                        metrics.accuracy_score(y_test, self.predict(X_test))))
            self.W = np.load('./immedia_parameter/{}_weight_{}_{}_{}_{}_{}.npy'.format(self.dataname,self.K_value, self.number_iter, self.inner_iter, self.learning_rate, self.seed))
            # self.W = W
            # print(W)
            test_predict.append(self.predict(X_test))
            print("epoch:", i, 'tr_l={:.3f} tr_a={:.5f}:'.format(self.loss_student(self.W, X_intercept, y),
                                                                 metrics.accuracy_score(y, self.predict(X))),
                  'te_l={:.3f} te_a={:.5f}:'.format(self.loss_student(self.W, X_test_intercept, y_test),
                                                    metrics.accuracy_score(y_test, self.predict(X_test))))
        # self.W = W
        np.save('./predict/{}_predict_{}_{}_{}_{}_{}.npy'.format(self.dataname, self.K_value, self.number_iter,
                                                                          self.inner_iter, self.learning_rate,
                                                                          self.seed), test_predict)

    def predict(self, X):
        return self.predict_cs(X)


    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return self.labels[np.argmax(self.W.dot(X_intercept.T), axis=0)]

    def predict_topk_acc(self, Y, X, kk_value=5):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        predict = self.W.dot(X_intercept.T)
        top_k_acc_list = np.argsort(predict, axis=0)[-kk_value:, :]
        count = 0
        for i in range(len(Y)):
            if Y[i] in top_k_acc_list[:,i]:
                count = count +1
        acc = count / len(Y)
        return acc


    def loss_student(self, W, X, y, C=1.0):
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
        loss_aug_inf = 1 - (self.labels[:, None] == y[None, :]) + np.matmul(W, np.transpose(X))  # (K, N)
        # sum over N of max value in loss_aug_inf
        loss_aug_inf_max_sum = np.sum(np.max(loss_aug_inf, axis=0))
        # sum over N of w_{y_i}^T * x_i
        wx_sum = np.sum(W[y] * X)
        multiclass_loss = C * (loss_aug_inf_max_sum - wx_sum)

        # total_loss = l2_loss + multiclass_loss
        total_loss = multiclass_loss / X.shape[0]
        return total_loss

    def grad_student(self, W, X, y, sub_gradient, C=1.0):
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

        # gradient of the other term
        # get the matrix of term 1 - delta(j, y_i) + w_j^T * x_i
        # loss_aug_inf = 1 - (self.labels[:, None] == y[None, :]) + np.matmul(W, np.transpose(X))  # (K, N)

        scores = np.matmul(W, np.transpose(X))
        loss_aug_inf = 1 + scores - scores[y, np.arange(X.shape[0])].reshape(1, X.shape[0])
        loss_aug_inf = (loss_aug_inf > 0) * loss_aug_inf
        # loss_aug_inf = 1 + np.matmul(W, np.transpose(X))

        # get the j_max that maximizes the above matrix for every sample
        # j_max = np.argmax(loss_aug_inf, axis=0)  # (N,)

        multiclass_grad_total = np.zeros((W.shape[0],W.shape[1]))

        for k in range(self.K_value+1):
            scores_inner = np.matmul(W, np.transpose(X))
            loss_modify = 1 + scores_inner - scores_inner[y, np.arange(X.shape[0])].reshape(1, X.shape[0])
            # loss_modify = (loss_modify > 0) * loss_modify
            # loss_modify = 1 + np.matmul(W, np.transpose(X))
            lamb = np.maximum(0, np.sort(loss_modify, axis=0)[::-1][self.K_value])
            hinge = loss_modify - lamb
            loss_modify[hinge<0]=0

            # j_max = np.argsort(loss_aug_inf, axis=0)[::-1][k]
            j_max = np.argsort(loss_aug_inf, axis=0)[::-1][k]

            # gradient of sum(...) is:   x_i, if k == j_max_i and k != y_i  (pos_case)
            #                           -x_i, if k != j_max_i and k == y_i  (neg_case)
            #                              0, otherwise
            pos_case = np.logical_and((self.labels[:, None] == j_max[None, :]), (self.labels[:, None] != y[None, :]))
            pos_case_2 = np.logical_and((loss_modify > 0), pos_case)

            # grad_condition_list = []
            #
            # for i in range(pos_case.shape[1]):
            #     if True in pos_case[:,i]:
            #         grad_condition_list.append(np.ones((pos_case.shape[0], ), dtype=bool))
            #     else:
            #         grad_condition_list.append(np.zeros((pos_case.shape[0],), dtype=bool))
            # grad_condition = np.transpose(np.asarray(grad_condition_list))

            grad_condition = np.logical_and((loss_modify > 0), pos_case)
            for i in range(grad_condition.shape[1]):
                if True in grad_condition[:, i]:
                    grad_condition[:, i] = np.ones((grad_condition.shape[0],), dtype=bool)
                else:
                    grad_condition[:, i] = np.zeros((grad_condition.shape[0],), dtype=bool)

            neg_case = np.logical_and((self.labels[:, None] != j_max[None, :]), (self.labels[:, None] == y[None, :]))
            neg_case = np.logical_and(grad_condition, neg_case)
            multiclass_grad = C * np.matmul(pos_case_2.astype(int) - neg_case.astype(int) , X)
            multiclass_grad_total = multiclass_grad_total + multiclass_grad

        # total_grad = l2_grad + (multiclass_grad_total/X.shape[0])-sub_gradient
        # total_grad = l2_grad + (multiclass_grad_total - sub_gradient)
        # total_grad = (1e-4)*l2_grad + multiclass_grad_total - sub_gradient
        total_grad = multiclass_grad_total - sub_gradient

        return total_grad

    def subgrad_student(self, W, X, y,C=1.0):
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

        # gradient of the other term
        # get the matrix of term 1 - delta(j, y_i) + w_j^T * x_i
        # loss_aug_inf = 1 - (self.labels[:, None] == y[None, :]) + np.matmul(W, np.transpose(X))  # (K, N)

        scores = np.matmul(W, np.transpose(X))
        loss_aug_inf = 1 + scores - scores[y, np.arange(X.shape[0])].reshape(1, X.shape[0])
        loss_aug_inf = (loss_aug_inf > 0) * loss_aug_inf
        # loss_aug_inf = 1 + np.matmul(W, np.transpose(X))
        # get the j_max that maximizes the above matrix for every sample
        # j_max = np.argmax(loss_aug_inf, axis=0)  # (N,)

        multiclass_grad_total = np.zeros((W.shape[0],W.shape[1]))

        for k in range(self.K_value):
            scores_inner = np.matmul(W, np.transpose(X))
            loss_modify = 1 + scores_inner - scores_inner[y, np.arange(X.shape[0])].reshape(1, X.shape[0])
            # loss_modify = 1 + np.matmul(W, np.transpose(X))
            # loss_modify = (loss_modify > 0) * loss_modify
            lamb = np.maximum(0, np.sort(loss_modify, axis=0)[::-1][self.K_value-1])
            hinge = loss_modify - lamb
            loss_modify[hinge<0]=0

            # j_max = np.argsort(loss_aug_inf, axis=0)[::-1][k]
            j_max = np.argsort(loss_aug_inf, axis=0)[::-1][k]

            # gradient of sum(...) is:   x_i, if k == j_max_i and k != y_i  (pos_case)
            #                           -x_i, if k != j_max_i and k == y_i  (neg_case)
            #                              0, otherwise
            pos_case = np.logical_and((self.labels[:, None] == j_max[None, :]), (self.labels[:, None] != y[None, :]))
            pos_case_2 = np.logical_and((loss_modify > 0), pos_case)

            # grad_condition_list = []
            #
            # for i in range(pos_case.shape[1]):
            #     if True in pos_case[:,i]:
            #         grad_condition_list.append(np.ones((pos_case.shape[0], ), dtype=bool))
            #     else:
            #         grad_condition_list.append(np.zeros((pos_case.shape[0],), dtype=bool))
            # grad_condition = np.transpose(np.asarray(grad_condition_list))

            grad_condition = np.logical_and((loss_modify > 0), pos_case)
            for i in range(grad_condition.shape[1]):
                if True in grad_condition[:, i]:
                    grad_condition[:, i] = np.ones((grad_condition.shape[0],), dtype=bool)
                else:
                    grad_condition[:, i] = np.zeros((grad_condition.shape[0],), dtype=bool)

            neg_case = np.logical_and((self.labels[:, None] != j_max[None, :]), (self.labels[:, None] == y[None, :]))
            neg_case = np.logical_and(grad_condition, neg_case)
            multiclass_grad = C * np.matmul(pos_case_2.astype(int) - neg_case.astype(int) , X)
            multiclass_grad_total = multiclass_grad_total + multiclass_grad

        total_grad = multiclass_grad_total

        return total_grad


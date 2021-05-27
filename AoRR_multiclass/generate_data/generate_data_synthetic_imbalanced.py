import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.datasets.samples_generator import make_blobs


X_train, y_train = make_blobs(n_samples=[5500, 500, 6000], centers=[(-6, -6), (0,0), (6,6)],
                       cluster_std=[1.0, 0.5, 1.0], random_state=100)
y_train[y_train==1] =0
y_train[y_train==2] =1
X_train = X_train[:, ::-1]
# np.save('datasets/Synthetic_data_train_val/data_train.npy',X_train)
# onehot_encoder = OneHotEncoder(sparse=False)
# y_train_new = y_train.reshape(len(y_train), 1)
# onehot_encoded = onehot_encoder.fit_transform(y_train_new).astype(int)
# np.save('datasets/Synthetic_data_train_val/label_train_noise_0.npy',onehot_encoded)


for ratio in [0.1, 0.2, 0.3, 0.4]:
    y_noise = np.copy(y_train)
    y_train_positive = np.argwhere(y_train==1)[:,0]
    selected_samples_index=np.asarray(random.sample(y_train_positive.tolist(), int(y_train_positive.shape[0] *ratio)))
    y_noise[selected_samples_index]=0
    onehot_encoder = OneHotEncoder(sparse=False)
    y_noise_new = y_noise.reshape(len(y_noise), 1)
    onehot_encoded = onehot_encoder.fit_transform(y_noise_new).astype(int)
    np.save('datasets/Synthetic_data_train_val/label_train_noise_{}.npy'.format(ratio),onehot_encoded)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_noise, s=20, cmap='bwr')
    plt.show()


X_val, y_val = make_blobs(n_samples=[500, 100, 600], centers=[(-6, -6), (0,0), (6,6)],
                       cluster_std=[1.0, 0.5, 1.0], random_state=0)
y_val[y_val==1] =0
y_val[y_val==2] =1
X_val = X_val[:, ::-1]
# np.save('datasets/Synthetic_data_train_val/data_val.npy',X_val)
# y_val_new = y_val.reshape(len(y_val), 1)
# onehot_encoded = onehot_encoder.fit_transform(y_val_new).astype(int)
# np.save('datasets/Synthetic_data_train_val/label_val.npy',onehot_encoded)




X_test, y_test = make_blobs(n_samples=[500, 100, 600], centers=[(-6, -6), (0,0), (6,6)],
                       cluster_std=[1.0, 0.5, 1.0], random_state=10000)
y_test[y_test==1] =0
y_test[y_test==2] =1
X_test = X_test[:, ::-1]
# np.save('datasets/Synthetic_data_train_val/data_test.npy',X_test)
# y_test_new = y_test.reshape(len(y_test), 1)
# onehot_encoded = onehot_encoder.fit_transform(y_test_new).astype(int)
# np.save('datasets/Synthetic_data_train_val/label_test.npy',onehot_encoded)





# kmeans = KMeans(10, random_state=0)
# labels = kmeans.fit(X).predict(X)

# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(y_true)
# print(integer_encoded)

# for ratio in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]:
#     y_noise = np.copy(y_true)
#     y_true_positive = np.argwhere(y_true==1)[:,0]
#     selected_samples_index=np.asarray(random.sample(y_true_positive.tolist(), int(y_true_positive.shape[0] *ratio)))
#     y_noise[selected_samples_index]=0
#     onehot_encoder = OneHotEncoder(sparse=False)
#     y_noise_new = y_noise.reshape(len(y_noise), 1)
#     onehot_encoded = onehot_encoder.fit_transform(y_noise_new).astype(int)
#     np.save('datasets/Synthetic_data/label1_noise_{}_train.npy'.format(ratio),onehot_encoded)
#     plt.scatter(X[:, 0], X[:, 1], c=y_noise, s=20, cmap='viridis')
#     plt.show()
#
#
# onehot_encoder = OneHotEncoder(sparse=False)
# y_true_new = y_true.reshape(len(y_true), 1)
# onehot_encoded = onehot_encoder.fit_transform(y_true_new).astype(int)

# np.save('datasets/Synthetic_data/data1_train.npy', X)
# np.save('datasets/Synthetic_data/label1_train.npy', onehot_encoded)
#




plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20, cmap='bwr')
plt.show()
plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, s=20, cmap='bwr')
plt.show()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, cmap='bwr')
plt.show()
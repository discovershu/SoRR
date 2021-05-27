import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random


from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=1000, centers=2,
                       cluster_std=0.4, random_state=0)
X = X[:, ::-1] # flip axes for better plotting

# kmeans = KMeans(10, random_state=0)
# labels = kmeans.fit(X).predict(X)

# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(y_true)
# print(integer_encoded)

for ratio in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]:
    y_noise = np.copy(y_true)
    y_true_positive = np.argwhere(y_true==1)[:,0]
    selected_samples_index=np.asarray(random.sample(y_true_positive.tolist(), int(y_true_positive.shape[0] *ratio)))
    y_noise[selected_samples_index]=0
    onehot_encoder = OneHotEncoder(sparse=False)
    y_noise_new = y_noise.reshape(len(y_noise), 1)
    onehot_encoded = onehot_encoder.fit_transform(y_noise_new).astype(int)
    np.save('datasets/Synthetic_data/label1_noise_{}_train.npy'.format(ratio),onehot_encoded)
    plt.scatter(X[:, 0], X[:, 1], c=y_noise, s=20, cmap='viridis')
    plt.show()


onehot_encoder = OneHotEncoder(sparse=False)
y_true_new = y_true.reshape(len(y_true), 1)
onehot_encoded = onehot_encoder.fit_transform(y_true_new).astype(int)

# np.save('datasets/Synthetic_data/data1_train.npy', X)
# np.save('datasets/Synthetic_data/label1_train.npy', onehot_encoded)
#




plt.scatter(X[:, 0], X[:, 1], c=y_true, s=20, cmap='viridis')
plt.show()
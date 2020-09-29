import time
import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn import metrics

from AoRR.hinge_DC import hinge_TOPK_DC


def sample_per_class(random_state, labels, size_ratio, forbidden_indices=None):
    uniqueValues, occurCount = np.unique(labels, return_counts=True)
    num_samples = len(labels)
    num_classes = len(uniqueValues)
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index] == class_index:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], int(len(sample_indices_per_class[class_index])*size_ratio), replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

Model_name = "Hinge"

Method = "TOPK_DC"


dataname = "monks-3"
kvalue = 70
kvalue2 =45
train_size = 0.5
val_size = 0.25
seed = 21


dataset = pd.read_csv('./dataset/Monk/{}.test'.format(dataname),header=None, delimiter=' ')
data = dataset.values[:,2:8].astype(float)
data = preprocessing.scale(data)
label = dataset.values[:,1].astype(np.int)

random_state = np.random.RandomState(seed)
uniqueValues, occurCount = np.unique(label, return_counts=True)
remaining_indices = list(range(len(label)))
train_indices = sample_per_class(random_state, label, train_size)
val_indices = sample_per_class(random_state, label, val_size*2, forbidden_indices=train_indices)
forbidden_indices = np.concatenate((train_indices, val_indices))
test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

X_train = data[train_indices,:]
y_train = label[train_indices]
y_train[y_train==0]=-1

X_val = data[val_indices,:]
y_val = label[val_indices]
y_val[y_val==0]=-1

X_test = data[test_indices, :]
y_test = label[test_indices]
y_test[y_test==0]=-1

mis_classification_rate = []

# for kvalue2 in range(1,kvalue):
if Method == "TOPK_DC":
    classifier = hinge_TOPK_DC(lr=0.01, num_iter=5, inner_iter=1000, k_value=kvalue, k2_value = kvalue2, seed = 1,dataname = dataname, Model_name=Model_name)  #### TopK DC case

t_start = time.time()
classifier.fit(X_train, y_train,X_val,y_val,X_test,y_test)
t_total = time.time() - t_start
print("total_time: ", t_total)

predicted, parameters = classifier.predict(X_test)

# np.save('./parameters/'+ Model_name+'_'+Method+'_'+ dataname + '.npy', parameters)

# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(y_test, predicted, digits=3)))

print("kvalue2:",kvalue2)
mis_classification_rate.append(
    (1-metrics.classification_report(y_test, predicted, digits=3, output_dict=True)['accuracy']))

print("miss_classification rate:{:.4f}".format(1-metrics.classification_report(y_test, predicted, digits=3, output_dict=True)['accuracy']))

# np.save('./real_results/{}_{}_missclassification_rate_for_k_prime.npy'.format(dataname, Model_name),np.asarray(mis_classification_rate))
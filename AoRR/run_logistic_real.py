import time
import numpy as np
import pandas as pd
from sklearn import preprocessing

from sklearn import datasets, svm, metrics

from AoRR.logisticregression_DC import LogisticRegression_TOPK_DC

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

Model_name = "LogisticRegression"

# Method = "Avg"
# Method = "ATK"
# Method = "Top1"
# Method = "TOPK_GD"
Method = "TOPK_DC"
# Method = "TOPK_DC_cvx"

dataname = "monks-3"
kvalue = 70
kvalue2 =20
train_size = 0.5
val_size = 0.25
seed = 2




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

X_val = data[val_indices,:]
y_val = label[val_indices]

X_test = data[test_indices, :]
y_test = label[test_indices]


mis_classification_rate = []


### classifier fit data
if Method == "TOPK_DC":
    classifier = LogisticRegression_TOPK_DC(lr=0.01, num_iter=10, inner_iter=1000, k_value = kvalue, k2_value = kvalue2,dataname = dataname, Model_name=Model_name) #### TopK DC case


t_start = time.time()
classifier.fit(X_train, y_train,X_val,y_val,X_test,y_test)
t_total = time.time()-t_start
print("total_time: ", t_total)

### classifier predict data

predicted, parameters = classifier.predict(X_test)

# np.save('./parameters/'+ Model_name+'_'+Method+'_'+ dataname + '.npy', parameters)
print("kvalue2:", kvalue2)
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(y_test, predicted, digits=3)))


mis_classification_rate.append((1-metrics.classification_report(y_test, predicted, digits=3, output_dict=True)['accuracy']))
print("miss_classification rate:{:.4f}".format(1-metrics.classification_report(y_test, predicted, digits=3, output_dict=True)['accuracy']))
# np.save('./real_results/{}_{}_missclassification_rate_for_k_prime.npy'.format(dataname, Model_name),np.asarray(mis_classification_rate))
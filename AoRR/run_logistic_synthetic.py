import time
import scipy.io
import numpy as np

from sklearn import datasets, svm, metrics

from AoRR.logisticregression_DC import LogisticRegression_TOPK_DC

Model_name = "LogisticRegression"

Method = "TOPK_DC"

dataname = "data4_1" ## or data6_1
kvalue = 2
mvalue =1


######read data

Toydata_path = './dataset/mydata/{}.mat'.format(dataname)

Toydata = scipy.io.loadmat(Toydata_path)

data = Toydata['x']

label = Toydata['y'].flatten()

uniqueValues, occurCount = np.unique(label, return_counts=True)
number_1 = uniqueValues[0]
number_2 = uniqueValues[1]
print ('We do binary classificaiton using number: ', number_1, 'and', number_2)

modified_label = [0 if x==number_1 else 1 for x in label]
modified_label = np.asarray(modified_label)

### classifier fit data

if Method == "TOPK_DC":
    classifier = LogisticRegression_TOPK_DC(lr=0.01, num_iter=10, inner_iter=1000, k_value = kvalue, k2_value = mvalue, seed =1234, dataname=dataname) #### TopK DC case

t_start = time.time()
classifier.fit(data, modified_label,data, modified_label,data, modified_label)
t_total = time.time()-t_start
print("total_time: ", t_total)

### classifier predict data

predicted, parameters = classifier.predict(data)

# np.save('./parameters/'+ Model_name+'_'+Method+'_'+ dataname + '.npy', parameters)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(modified_label, predicted, digits=3)))


import scipy.io
import numpy as np

from sklearn import metrics

from AoRR.hinge_DC import hinge_TOPK_DC

Model_name = "Hinge"

Method = "TOPK_DC"

dataname = "data4_1" ## or data6_1
kvalue = 2
mvalue =1


######read data

Toydata_path = './dataset/mydata/{}.mat'.format(dataname)

Toydata = scipy.io.loadmat(Toydata_path)

data = Toydata['x']
label = Toydata['y'].flatten()

if Method == "TOPK_DC":
    classifier = hinge_TOPK_DC(lr=0.01, num_iter=5, inner_iter=1000, k_value=kvalue, k2_value = mvalue, seed =1234, dataname=dataname)  #### TopK DC case

classifier.fit(data, label,data, label,data, label)

predicted, parameters = classifier.predict(data)

# np.save('./parameters/'+ Model_name+'_'+Method+'_'+ dataname + '.npy', parameters)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(label, predicted, digits=3)))
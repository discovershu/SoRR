import matplotlib.pyplot as plt
import scipy.io
import numpy as np


######read data
dataname = "data6_1"

Toydata_path = './dataset/mydata/{}.mat'.format(dataname)

Toydata = scipy.io.loadmat(Toydata_path)

# Model_name = "LogisticRegression"
Model_name = "Hinge"

Method1 = "Avg"
Method2 = "ATK"
Method3 = "Top1"
Method4 = "TOPK_DC"


def plot_toy(Toydata, x_values, y_values1, y_values2, y_values3, y_values4):
    data = Toydata['x']
    label = Toydata['y']
    uniqueValues, occurCount = np.unique(label, return_counts=True)
    number_1 = uniqueValues[0]
    number_2 = uniqueValues[1]

    number_1_index = np.where((label == number_1) == True)[0]
    number_2_index = np.where((label == number_2) == True)[0]

    plt.scatter(data[number_1_index, 0], data[number_1_index, 1], s=30, marker='o', facecolors='none',
                edgecolors='b')
    plt.scatter(data[number_2_index, 0], data[number_2_index, 1], s=30, marker='x', color='r')
    plt.plot(x_values, y_values1, label='Average', linestyle=':', linewidth=2)
    plt.plot(x_values, y_values2, label='AT$_{k=2}$', linestyle='-.', linewidth=2)
    plt.plot(x_values, y_values3, label='Maximum', linestyle='--', linewidth=2)
    plt.plot(x_values, y_values4, label='AoRR', linestyle='-', linewidth=2)
    plt.ylim((-1, 3))
    plt.xlim((-1, 3))
    # plt.tick_params(labelsize=20)
    plt.legend(prop={'size': 12, 'weight':'bold'})
    plt.title(r'$\bf{Classification}$ $\bf{Boundary}$', fontsize=15)
    # plt.title('Classification Boundary')
    # plt.savefig('./fig/{}_{}_Classification_Boundary.png'.format(Model_name, dataname), dpi=800)
    plt.show()


### plot decision boundary

x_values = np.arange(-1, 3.1, 0.1)

parameter1 = np.load('./parameters/{}_{}_{}.npy'.format(Model_name,Method1,dataname))
plot_y1 = -(parameter1[0] + np.dot(parameter1[1], x_values)) / parameter1[2]

parameter2 = np.load('./parameters/{}_{}_{}.npy'.format(Model_name,Method2,dataname))
plot_y2 = -(parameter2[0] + np.dot(parameter2[1], x_values)) / parameter2[2]

parameter3 = np.load('./parameters/{}_{}_{}.npy'.format(Model_name,Method3,dataname))
plot_y3 = -(parameter3[0] + np.dot(parameter3[1], x_values)) / parameter3[2]

parameter4 = np.load('./parameters/{}_{}_{}.npy'.format(Model_name,Method4,dataname))
plot_y4 = -(parameter4[0] + np.dot(parameter4[1], x_values)) / parameter4[2]




plot_toy(Toydata, x_values, plot_y1, plot_y2, plot_y3, plot_y4)


import matplotlib.pyplot as plt
import numpy as np


######read data
dataname = "monks-3" #phoneme, monks-3, australian, splice

hinge = np.load('./real_results/{}_Hinge_missclassification_rate_for_k_prime.npy'.format(dataname))
logistic = np.load('./real_results/{}_LogisticRegression_missclassification_rate_for_k_prime.npy'.format(dataname))

plt.xlim((0,70))
plt.ylim((0.08, 0.4))#monks, australian
# plt.ylim((0.14, 0.4))#splice
# plt.ylim((0.21, 0.4))#phoneme
x = range(1,len(hinge)+1)
# x = range(len(hinge))
# line1, =plt.plot(x,logistic,label='AoRR + Logistic Loss', color='b')
line2, =plt.plot(x,hinge,label='AoRR + Hinge Loss', color='r')
###Monk-3
# line3 =plt.axhline(y=0.1676, color='g', linestyle='--', label='AT$_k$ + Logistic Loss',linewidth=2)
line4 =plt.axhline(y=0.1704, color='y', linestyle='--', label='AT$_k$ + Hinge Loss',linewidth=2)
# line5 =plt.axhline(y=0.2046, color='c', linestyle='-.', label='Average + Logistic Loss',linewidth=2)
line6 =plt.axhline(y=0.1861, color='k', linestyle='-.', label='Average + Hinge Loss',linewidth=2)
# line7 =plt.axhline(y=0.2241, color='m', linestyle=':', label='Maximum + Logistic Loss',linewidth=2)
line8 =plt.axhline(y=0.2204, color='brown', linestyle=':', label='Maximum + Hinge Loss',linewidth=2)

####Australian
# # line3 =plt.axhline(y=0.117, color='g', linestyle='--', label='AT$_k$ + Logistic Loss',linewidth=2)
# line4 =plt.axhline(y=0.1251, color='y', linestyle='--', label='AT$_k$ + Hinge Loss',linewidth=2)
# # line5 =plt.axhline(y=0.1427, color='c', linestyle='-.', label='Average + Logistic Loss',linewidth=2)
# line6 =plt.axhline(y=0.1474, color='k', linestyle='-.', label='Average + Hinge Loss',linewidth=2)
# # line7 =plt.axhline(y=0.1988, color='m', linestyle=':', label='Maximum + Logistic Loss',linewidth=2)
# line8 =plt.axhline(y=0.1982, color='brown', linestyle=':', label='Maximum + Hinge Loss',linewidth=2)

####splice
# # line3 =plt.axhline(y=0.1612, color='g', linestyle='--', label='AT$_k$ + Logistic Loss',linewidth=2)
# line4 =plt.axhline(y=0.1623, color='y', linestyle='--', label='AT$_k$ + Hinge Loss',linewidth=2)
# # line5 =plt.axhline(y=0.1725, color='c', linestyle='-.', label='Average + Logistic Loss',linewidth=2)
# line6 =plt.axhline(y=0.1625, color='k', linestyle='-.', label='Average + Hinge Loss',linewidth=2)
# # line7 =plt.axhline(y=0.2357, color='m', linestyle=':', label='Maximum + Logistic Loss',linewidth=2)
# line8 =plt.axhline(y=0.234, color='brown', linestyle=':', label='Maximum + Hinge Loss',linewidth=2)

####Phoneme
# # line3 =plt.axhline(y=0.2417, color='g', linestyle='--', label='AT$_k$ + Logistic Loss',linewidth=2)
# line4 =plt.axhline(y=0.2288, color='y', linestyle='--', label='AT$_k$ + Hinge Loss',linewidth=2)
# # line5 =plt.axhline(y=0.255, color='c', linestyle='-.', label='Average + Logistic Loss',linewidth=2)
# line6 =plt.axhline(y=0.2288, color='k', linestyle='-.', label='Average + Hinge Loss',linewidth=2)
# # line7 =plt.axhline(y=0.2867, color='m', linestyle=':', label='Maximum + Logistic Loss',linewidth=2)
# line8 =plt.axhline(y=0.2881, color='brown', linestyle=':', label='Maximum + Hinge Loss',linewidth=2)

plt.xlabel(r'$\bf{m}$ $\bf{value}$', fontsize=20)
plt.ylabel(r'$\bf{Misclassification}$ $\bf{Rates}$ $\bf{(\%)}$', fontsize=20)
plt.title('{}'.format(r'$\bf{Monk}$ $\bf{(k=70)}$'), fontsize=20)
# plt.title('{}'.format(r'$\bf{Australian}$ $\bf{(k=80)}$'), fontsize=20)
# plt.title('{}'.format(r'$\bf{Splice}$ $\bf{(k=450)}$'), fontsize=20)
# plt.title('{}'.format(r'$\bf{Phoneme}$ $\bf{(k=1400)}$'), fontsize=20)
# first_legend = plt.legend(handles=[line1, line3, line5, line7], prop={'size': 15, 'weight':'bold'}, loc='upper right')
# ax = plt.gca().add_artist(first_legend)
plt.legend(handles=[line2, line4, line6, line8], prop={'size': 15, 'weight':'bold'}, loc='upper right')
# plt.show()
plt.savefig('./fig/{}_misclassification_rates.png'.format(dataname), dpi=800)
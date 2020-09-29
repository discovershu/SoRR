import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

x = np.linspace(-1.5, 2.0, 1001, dtype=float)

# y_logistics = (1 + np.exp(-x))
y_logistics = np.log((1 + np.exp(-x))) / np.log(2)
y_01 = x < 0
y_hinge = 1.0 - x
y_hinge[y_hinge < 0] = 0
y_hinge_ti = 1.0-x
y_hinge_ti[y_hinge_ti < 0] = 0
y_hinge_ti_1 = y_hinge_ti -0.4#0.4
y_hinge_ti_2 = y_hinge_ti -1.3#1.3
y_hinge_ti_1[y_hinge_ti_1 < 0] = 0
y_hinge_ti_2[y_hinge_ti_2 < 0] = 0
y_hinge_ti_final = y_hinge_ti_1 - y_hinge_ti_2

# y_logistics_ti = np.log((1 + np.exp(-x))) / np.log(2)
y_logistics_ti = np.log2(1 + np.exp(-x))
y_logistics_ti_0 = y_logistics_ti -0.3
y_logistics_ti_1 = y_logistics_ti -0.4#0.4
y_logistics_ti_2 = y_logistics_ti -1.3#1.2
y_logistics_ti_0[y_logistics_ti_0 < 0] = 0
y_logistics_ti_1[y_logistics_ti_1 < 0] = 0
y_logistics_ti_2[y_logistics_ti_2 < 0] = 0
y_logistics_ti_final = y_logistics_ti_1 - y_logistics_ti_2


# plt.figure(figsize=(5,5),facecolor='w')
plt.plot(x,y_01,'k-',label = r'$\bf{01}$',lw = 2)

plt.plot(x,y_logistics,'b-',label=r'$\bf{Logistic}$',lw =2)
plt.plot(x,y_logistics_ti_0,'b:',label=r'AT$_{k}$',lw =2)
plt.plot(x,y_logistics_ti_final,'r--',label='AoRR',lw =2)

# plt.plot(x,y_hinge,'b-',label = '$\ell_{hinge}(z)=[1-z]_+$',lw = 2)
# plt.plot(x,y_hinge_ti_1,'b:',label = '$[\ell_{hinge}(z)-0.4]_+$',lw = 2)
# plt.plot(x,y_hinge_ti_final,'b--',label = '$[\ell_{hinge}(z)-0.4]_+ -[\ell_{hinge}(z)-1.2]_+$',lw = 2)

# plt.grid()
# plt.title('常见损失函数', fontsize=16)
plt.legend(prop={'size': 15, 'weight':'bold'}, loc='center right')
plt.fill_between(x, -0.1, 3, where=(x>=0), color='lightgrey')
plt.xlabel(r'$\bf{yf(x)}$', fontsize=15)
plt.ylabel(r'$\bf{Loss}$', fontsize=15)
plt.xlim((-1.0, 2.0))
plt.ylim((-0.1, 2.0))
plt.text(-0.8, 1.8, r'$\bf{yf(x)<0}$', fontsize=20)
plt.text(0.7, 1.8, r'$\bf{yf(x)>0}$', fontsize=20)
plt.hlines(0.0, -1.0, 2.0, colors = "k",lw = 0.5)
plt.savefig('fig/interpretation_aggregate_loss.png', dpi=800)
# plt.show()

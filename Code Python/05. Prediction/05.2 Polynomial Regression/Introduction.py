import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

dx = 0.5
x = np.arange(-6, 7, dx)
x = x.reshape(len(x), 1)

y = np.power(x, 3) + 50*np.random.randn(x.shape[0], 1) + 50
y = y.reshape(len(y), 1)

x1 = np.append(np.ones((x.shape[0], 1)), x, axis=1)
x2 = np.append(x1, np.power(x,2), axis=1)
x3 = np.append(x2, np.power(x,3), axis=1)
x10 = np.append(x3, np.power(x,4), axis=1)
x10 = np.append(x10,np.power(x,5), axis=1)
x10 = np.append(x10,np.power(x,6), axis=1)
x10 = np.append(x10,np.power(x,7), axis=1)
x10 = np.append(x10,np.power(x,8), axis=1)
x10 = np.append(x10,np.power(x,9), axis=1)
x10 = np.append(x10,np.power(x,10), axis=1)



data = [x1, x2, x3, x10]
tag = ['Linear Regression', 'Quadratic Regression', 'Cubic Regression', 'Decic Regression']

plt.figure()
for i in range(0,4):
    plt.subplot(2,2,i+1)
    plt.title(tag[i])
    plt.scatter(x, y)
    [U,S,V] = np.linalg.svd(data[i], full_matrices=False)
    c = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), y])
    plt.plot(x, data[i].dot(c), 'r')
    plt.xlim([-6, 7])
    plt.legend(['Data','Regression'])
plt.show()

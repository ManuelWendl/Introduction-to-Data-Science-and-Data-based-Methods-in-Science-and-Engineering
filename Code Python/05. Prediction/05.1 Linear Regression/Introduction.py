import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

dx = 0.1
x = np.transpose(np.arange(-2, 2, dx))
x = x.reshape((len(x), 1))
d = np.ones(((x.shape[0]), (x.shape[1]+1)))
d[:, 1:] = x
y = 3*x + 3*np.random.randn(40, 1) + 4

# Pseudo Inverse
[U, S, V] = np.linalg.svd(d, full_matrices=False)
m = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), y])


plt.figure()
plt.scatter(x, y)
plt.plot(x, d.dot(m), color='r')
plt.plot(x, 3*x+4, color='b', linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Data', 'Linear Regression', 'True Fit'])
plt.show()


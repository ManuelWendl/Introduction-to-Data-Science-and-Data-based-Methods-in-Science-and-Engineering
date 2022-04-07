import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

imp = np.genfromtxt('atmosphere.csv', delimiter=',')
# contents:
#   #1    altitude
#   #2    pressure
#   #3    temperature
#   #4    humidity

# target is to predict the pressure from the

data = imp[:, [0, 2, 3]]
target = imp[:, 1]


[U, S, V] = np.linalg.svd(data, full_matrices=False)
m = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), np.matrix.transpose(target)])

plt.figure()
plt.plot(np.arange(len(target)), target, color='k', linestyle='-', marker='o')
plt.plot(np.arange(len(target)), np.dot(data, m), color='r', linestyle='-', marker='o')
plt.title('Non linearized Prediction')
plt.xlabel('datasets')
plt.ylabel('pressure')
plt.legend(['True Pressure','Predicted Pressure'])

cap=['altitude','temperature','humidity']

plt.figure()
for i in range(0, 3):
    plt.subplot(1,3,i+1)
    plt.plot(data[:,i],target,'o-')
    plt.xlabel(cap[i])
    plt.ylabel('pressure')


data[:, 0] = np.log(data[:, 0]+1)
print(data)

[U, S, V] = np.linalg.svd(data, full_matrices=False)
m = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), np.matrix.transpose(target)])

plt.figure()
plt.plot(np.arange(len(target)), target, color='k', linestyle='-', marker='o')
plt.plot(np.arange(len(target)), np.dot(data,m), color='r', linestyle='-', marker='o')
plt.title('Linearized Prediction')
plt.xlabel('datasets')
plt.ylabel('pressure')
plt.legend(['True Pressure', 'Predicted Pressure'])
plt.show()

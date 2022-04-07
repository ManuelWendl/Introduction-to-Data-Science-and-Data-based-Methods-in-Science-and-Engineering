import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

imp = np.genfromtxt('05. Prediction/05.2 Polynomial Regression/yacht_hydrodynamics.csv', delimiter=',')

# Contents:
# 1. Longitudinal position of the center of buoyancy, adimensional.
# 2. Prismatic coefficient, adimensional.
# 3. Length-displacement ratio, adimensional.
# 4. Beam-draught ratio, adimensional.
# 5. Length-beam ratio, adimensional.
# 6. Froude number, adimensional.
# The measured variable is the residuary resistance per unit weight of displacement:
# 7. Residuary resistance per unit weight of displacement, adimensional.

data = np.ones((imp.shape[0], imp.shape[1]))
data[:, 1:] = imp[:, 0:6]
target = imp[:, 6]

traindata = data[1:196, :]
traintarget = target[1:196]

testdata = data[196:, :]
testtarget = target[196:]

# Linear Model

[U, S, V] = np.linalg.svd(traindata, full_matrices=False)
c = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), traintarget])

plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.plot(testtarget, color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata, c), color='r', linestyle='-', marker='o')
plt.title('Linear regression model')
plt.legend(['True residuary resistance', 'Predicted residuary resistance'])

ind = np.argsort(testtarget)
plt.subplot(2, 3, 4)
plt.plot(testtarget[ind], color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata[ind,:], c), color='r', linestyle='-', marker='o')
plt.title('Linear regression model sorted')
plt.legend(['True residuary resistance', 'Predicted residuary resistance'])

# Quadratic Model

data = np.append(data, imp[:, 0:6]**2, axis=1)
traindata = data[1:196, :]
testdata = data[196:, :]

[U, S, V] = np.linalg.svd(traindata, full_matrices=False)
c = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), traintarget])

plt.subplot(2, 3, 2)
plt.plot(testtarget, color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata, c), color='r', linestyle='-', marker='o')
plt.title('Quadratic regression model')
plt.legend(['True residuary resistance', 'Predicted residuary resistance'])

ind = np.argsort(testtarget)
plt.subplot(2, 3, 5)
plt.plot(testtarget[ind], color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata[ind, :], c), color='r', linestyle='-', marker='o')
plt.title('Quadratic regression model sorted')
plt.legend(['True residuary resistance', 'Predicted residuary resistance'])

# Cubic Model

data = np.append(data, imp[:, 0:6]**3, axis=1)
traindata = data[1:196, :]
testdata = data[196:, :]

c = np.linalg.pinv(traindata).dot(traintarget)

plt.subplot(2, 3, 3)
plt.plot(testtarget, color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata, c), color='r', linestyle='-', marker='o')
plt.title('Cubic regression model')
plt.legend(['True residuary resistance', 'Predicted residuary resistance'])

ind = np.argsort(testtarget)
plt.subplot(2, 3, 6)
plt.plot(testtarget[ind], color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata[ind,:], c), color='r', linestyle='-', marker='o')
plt.title('Cubic regression model sorted')
plt.legend(['True residuary resistance', 'Predicted residuary resistance'])
plt.show()
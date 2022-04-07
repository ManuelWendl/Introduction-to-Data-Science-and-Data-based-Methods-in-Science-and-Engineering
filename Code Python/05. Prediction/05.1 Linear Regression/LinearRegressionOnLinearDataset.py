import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

imp = np.genfromtxt('Car_dataset.csv', delimiter=',')

# Contents of Data
#   #1 : mile per galon (fule efficiency)
#   #2 : cylinders
#   #3 : engine displacement in cubic inches
#   #4 : horsepower
#   #5 : weight
#   #6 : acceleration
#   #7 : year of model
#   #8 : origin (1:America, 2:European, 3: Japanese)

d = imp[:, 1:7]

d[np.isnan(d)] = 0

data = np.ones((d.shape[0], d.shape[1]+1))
data[:, 1:] = d

target = imp[:, 0]

traindata = data[0::2, :]
traintarget = target[0::2]

testdata = data[1::2, :]
testtarget = target[1::2]

[U, S, V] = np.linalg.svd(traindata, full_matrices=False)
m = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), np.matrix.transpose(traintarget)])

fig = plt.figure(figsize=(20, 10))
sp1 = fig.add_subplot(221)
plt.plot(traintarget, color='k', linestyle='-', marker='o')
plt.plot(np.dot(traindata,m), color='r', linestyle='-', marker='o')
plt.title('Prediction trainings dataset')
plt.legend(['True efficiency', 'Predicted efficiency'])

ind = np.argsort(traintarget)
traintarget_sorted = traintarget[ind]

sp2 = fig.add_subplot(223)
plt.plot(traintarget_sorted, color='k', linestyle='-', marker='o')
plt.plot(np.dot(traindata[ind, :],m), color='r', linestyle='-', marker='o')
plt.title('Prediction trainings dataset sorted')
plt.legend(['True efficiency', 'Predicted efficiency'])

# Prediction on Testdata
sp3 = fig.add_subplot(222)
plt.plot(testtarget, color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata, m), color='r', linestyle='-', marker='o')
plt.title('Prediction test dataset')
plt.legend(['True efficiency','Predicted efficiency'])

ind = np.argsort(testtarget)
testtarget_sorted = testtarget[ind]

sp4 = fig.add_subplot(224)
plt.plot(testtarget_sorted,color='k', linestyle='-', marker='o')
plt.plot(np.dot(testdata[ind, :],m), color='r', linestyle='-', marker='o')
plt.title('Prediction test dataset sorted')
plt.legend(['True efficiency','Predicted efficiency'])
plt.show()

data = data[:,1:8]
std_data = np.ma.std(data,0)
std_data = std_data.reshape(-1, 1)
mean_data = np.ma.mean(data,axis=0)
mean_data = mean_data.reshape(-1, 1)
print(mean_data)

deviation_data = data-np.dot(np.ones((data.shape[0], 1)), np.matrix.transpose(mean_data))

std_deviation_data = deviation_data/np.dot(np.ones((data.shape[0],1)),np.matrix.transpose(std_data))

[U,S,V] = np.linalg.svd(std_deviation_data,full_matrices=False)
m = reduce(np.dot, [np.matrix.transpose(V), np.linalg.inv(np.diag(S)), np.matrix.transpose(U), np.matrix.transpose(target)])


plt.figure(figsize=(20,10))
plt.bar(np.arange(0,len(m)), m)
cat = ['cylinders','engine displacement', 'weight', 'horsepower', 'acceleration', 'year of model', 'origin']
plt.xticks(np.arange(len(cat)), cat)
plt.title('Influence of individual factors on fuel efficiency (linear model)')
plt.xlabel('Factors')
plt.ylabel('Correlation')
plt.show()
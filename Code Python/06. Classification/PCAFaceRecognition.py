import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# Import Data
m = 192
n = 168

allfaces = np.zeros((32256, 2220))

for i in range(1, 38):
    for ii in range(1, 61):
        if i < 10:
            url = 'YaleFaces/yaleB0' + str(i) + '/' + 'Picture_' + str(ii) + '.pgm'
            face = imread(url)
            allfaces[:, (i - 1) * 60 + ii - 1] = face.reshape(n * m)
        else:
            url = 'YaleFaces/yaleB' + str(i) + '/' + 'Picture_' + str(ii) + '.pgm'
            face = imread(url)
            allfaces[:, (i - 1) * 60 + ii - 1] = face.reshape(n * m)

# Visualization Plots
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
firstimage = np.ones((6 * m, 6 * n))
count = 0
for i in range(1, 7):
    for ii in range(1, 7):
        firstimage[(i - 1) * m:i * m, (ii - 1) * n:ii * n] = allfaces[:, count * 60].reshape(m, n)
        count = count + 1

plt.imshow(firstimage, cmap='gray')

plt.subplot(1, 2, 2)
lightings = np.ones((7 * m, 8 * n))
count = 0
for i in range(1, 8):
    for ii in range(1, 9):
        lightings[(i - 1) * m:i * m, (ii - 1) * n:ii * n] = allfaces[:, count].reshape(m, n)
        count = count + 1

plt.imshow(lightings, cmap='gray')

# Keep test images (2 of every person)
testfaces = np.empty((32256, 74))

for i in range(0, 37):
    ind = np.random.randint(61, size=2)
    testfaces[:, [i*2, i*2 + 1]] = allfaces[:, i * 58 + ind]
    allfaces = np.delete(allfaces, i * 58 + ind, 1)

X = allfaces
print(X.shape)

# Average Face
M = np.ma.mean(X, 1)
plt.figure()
plt.imshow(M.reshape(m, n), cmap='gray')
plt.title('Average Face')
plt.show()

M = M.reshape((32256, 1))
B = X - M @ np.ones((1, X.shape[1]))

# SVD and eigen basis
U, S, V = np.linalg.svd(B, full_matrices=False)

# Principal Components in columns of U
PC = U
# Principal components in V (B*V*S reconstruction)
# PC = B*V*S

PCs = np.ones((5 * m, 5 * n))
count = 0
for i in range(1, 6):
    for ii in range(1, 6):
        PCs[(i - 1) * m:i * m, (ii - 1) * n:ii * n] = PC[:, count].reshape(m, n)
        count = count + 1

plt.figure(figsize=(10, 10))
plt.imshow(PCs, cmap='gray')
plt.title('First 25 principal components')
plt.show()

# Projection onto Principal Components

person1 = 2
person2 = 8

images1 = allfaces[:, (person1 - 1) * 58:(person1 - 1) * 58 + 57]
images2 = allfaces[:, (person2 - 1) * 58:(person2 - 1) * 58 + 57]

plt.figure(figsize=(20, 20))
plt.subplot(1, 3, 1)
images1m = np.ones((8 * m, 7 * n))
count = 0
for i in range(1, 9):
    for ii in range(1, 8):
        images1m[(i - 1) * m:i * m, (ii - 1) * n:ii * n] = images1[:, count].reshape(m, n)
        count = count + 1

plt.imshow(images1m, cmap='gray')
plt.title('Pictures of person ' + str(person1))

plt.subplot(1, 3, 3)
images2m = np.ones((8 * m, 7 * n))
count = 0
for i in range(1, 9):
    for ii in range(1, 8):
        images2m[(i - 1) * m:i * m, (ii - 1) * n:ii * n] = images2[:, count].reshape(m, n)
        count = count + 1

plt.imshow(images2m, cmap='gray')
plt.title('Pictures of person ' + str(person2))

PCchosen = [4, 5, 6]

Proj1 = (PC[:, PCchosen]).T.dot(images1)
Proj2 = (PC[:, PCchosen]).T.dot(images2)

ax = plt.subplot(1, 3, 2, projection='3d')
plt.plot(Proj1[0, :], Proj1[1, :], Proj1[2, :], 'rd', linewidth=4)
plt.plot(Proj2[0, :], Proj2[1, :], Proj2[2, :], 'bd', linewidth=4)
ax.set_xlabel('Principal Component ' + str(PCchosen[0]))
ax.set_ylabel('Principal Component ' + str(PCchosen[1]))
ax.set_zlabel('Principal Component ' + str(PCchosen[2]))
plt.grid('on')
plt.legend(['Person ' + str(person1), 'Person ' + str(person2)])
plt.show()

# Person Mean Face
meanFaces = np.empty((32256, 37))
for i in range(0, 37):
    meanFaces[:, i] = np.ma.mean(X[:, i*58:i*58+57], 1)

plt.figure(figsize=(10, 10))
meanFacesm = np.ones((6 * m, 6 * n))
count = 0
for i in range(1, 7):
    for ii in range(1, 7):
        meanFacesm[(i - 1) * m:i * m, (ii - 1) * n:ii * n] = meanFaces[:, count].reshape(m, n)
        count = count + 1

plt.imshow(meanFacesm, cmap='gray')
plt.title('Mean face of each person')
plt.show()

# Choose Principal Components for Prediction
plt.figure(figsize=(20, 20))
plt.subplot(2, 2, 1)
plt.semilogy(np.arange(1, len(S)+1), S, '-o')
plt.xlim([0, 400])
plt.title('Singular Values sigma')

plt.subplot(2, 2, 2)
lmbda = S ** 2
var = np.empty(len(lmbda))
for i in range(0, len(lmbda)):
    var[i] = np.ma.sum(lmbda[1:i]) / np.ma.sum(lmbda)
plt.plot(var, '-o')
plt.xlim([0, 400])
plt.title('Reached variance with k sigma')

PCchosen1 = [0, 1, 2]
Proj1 = (PC[:, PCchosen1]).T.dot(meanFaces)
ax = plt.subplot(2, 2, 3, projection='3d')
plt.plot(Proj1[0, :], Proj1[1, :], Proj1[2, :], 'rd', linewidth=4)
ax.set_xlabel('Principal Component ' + str(PCchosen1[0]))
ax.set_ylabel('Principal Component ' + str(PCchosen1[1]))
ax.set_xlabel('Principal Component ' + str(PCchosen1[2]))

PCchosen2 = np.arange(5, 25, 1)
Proj = (PC[:, PCchosen2]).T.dot(meanFaces)
ax = plt.subplot(2, 2, 4, projection='3d')
plt.plot(Proj[0, :], Proj[1, :], Proj[2, :], 'rd', linewidth=4)
ax.set_xlabel('Principal Component ' + str(PCchosen1[0]))
ax.set_ylabel('Principal Component ' + str(PCchosen1[1]))
ax.set_xlabel('Principal Component ' + str(PCchosen1[2]))
plt.show()

# Face Recognition
Testpicture = 10

ProjTest = (PC[:, PCchosen2]).T.dot(testfaces[:, Testpicture])
ProjTest = ProjTest.reshape(20, 1)

ProjDiff = Proj - ProjTest.dot(np.ones((1, 37)))

ProjDiff = sum(abs(ProjDiff), 0)
ind = np.argmin(ProjDiff)

plt.figure(figsize=(10,10))
plt.subplot(1, 3, 1)
plt.imshow(testfaces[:, Testpicture].reshape(m,n), cmap='gray')
plt.title('Testimage of Person: '+str(round(Testpicture / 2)))
plt.subplot(1, 3, 2)
plt.imshow(meanFaces[:, ind].reshape(m, n), cmap='gray')
plt.title('Recognized Person: ' + str(ind))
ax = plt.subplot(1, 3, 3, projection='3d')
images = allfaces[:, (round(Testpicture / 2) - 1) * 58:(round(Testpicture / 2) - 1) * 58 + 57]
Projp = (PC[:, PCchosen]).T.dot(images)
plt.plot(Projp[0, :], Projp[1, :], Projp[2, :], 'kd', linewidth=4)
plt.plot(ProjTest[0, :], ProjTest[1, :], ProjTest[2, :], 'rd', linewidth=4)
ax.set_xlabel('Principal Component ' + str(PCchosen1[0]))
ax.set_ylabel('Principal Component ' + str(PCchosen1[1]))
ax.set_xlabel('Principal Component ' + str(PCchosen1[2]))
plt.show()

count = 0
for i in range(0, 74):
    ProjTest = (PC[:, PCchosen2]).T.dot(testfaces[:, i])
    ProjTest = ProjTest.reshape(20, 1)
    ProjDiff = Proj - ProjTest.dot(np.ones((1,37)))
    ProjDiff = sum(abs(ProjDiff), 1)
    ind = np.argmin(ProjDiff)
    if ind == round(i / 2):
        count = count + 1
print('Performance of the recognition algorithm: %f' + str(count / 74))

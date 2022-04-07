import numpy as np
from matplotlib.image import imread
from PIL import Image
import matplotlib.pyplot as plt
import cvxpy as cvx

# Import Data
m = 192
n = 168

allfaces = np.zeros((32256, 2220))

for i in range(1, 38):
    for ii in range(1, 61):
        if i < 10:
            url = '/Users/manuelwendl/Dokumente/01_Universität/TUM Skripte und Bücher/Data Science Introduction in Science/Code Python/06. Classification/YaleFaces/yaleB0' + str(i) + '/' + 'Picture_' + str(ii) + '.pgm'
            face = imread(url)
            allfaces[:, (i - 1) * 60 + ii - 1] = face.reshape(n * m)
        else:
            url = '/Users/manuelwendl/Dokumente/01_Universität/TUM Skripte und Bücher/Data Science Introduction in Science/Code Python/06. Classification/YaleFaces/yaleB' + str(i) + '/' + 'Picture_' + str(ii) + '.pgm'
            face = imread(url)
            allfaces[:, (i - 1) * 60 + ii - 1] = face.reshape(n * m)

# Keep test images (2 of every person)
testfaces = np.empty((32256, 74))

for i in range(0, 37):
    ind = np.random.randint(61, size=2)
    testfaces[:, [i * 2, i * 2 + 1]] = allfaces[:, i * 58 + ind]
    allfaces = np.delete(allfaces, i * 58 + ind, 1)

X = allfaces

# Build Basis PSI
o = 15
p = 13

size = p, o

# Basis Matrix PSI

PSI = np.empty((195, X.shape[1]))

for i in range(X.shape[1]):
    img = X[:, i].reshape(m, n)
    img = Image.fromarray(img)
    img.thumbnail(size, Image.ANTIALIAS)
    img = np.array(img)
    PSI[:, i] = img.reshape(o * p)

# Reshape testfaces

b = np.empty((195, 74))
for i in range(testfaces.shape[1]):
    img = testfaces[:, i].reshape(m, n)
    img = Image.fromarray(img)
    img.thumbnail(size, Image.ANTIALIAS)
    img = np.array(img)
    b[:, i] = img.reshape(o * p)

# Recognition comparison L1, L2
num = 21
# L2
xL2 = np.linalg.pinv(PSI) @ b[:, num]
# L1
xL1 = cvx.Variable(PSI.shape[1])
objective = cvx.Minimize(cvx.norm(xL1, 1))
constraints = [cvx.norm(PSI @ xL1 - b[:, num], 2) <= .01]
prob = cvx.Problem(objective=objective, constraints=constraints)
prob.solve()
xL1 = xL1.value

plt.figure(figsize=(20, 20))
plt.subplot(4, 2, 1)
plt.imshow(testfaces[:, num].reshape(m, n), cmap='gray')
plt.title('Test Picture')
plt.subplot(4, 2, 2)
plt.imshow(b[:, num].reshape(o, p), cmap='gray')
plt.title('Downscaled Picture')
plt.subplot(4, 2, 3)
plt.title('x least square')
plt.plot(xL2)
plt.xlim([1, 2146])
plt.subplot(4, 2, 4)
plt.title('x sparse (L_1 norm)')
plt.plot(xL1)
plt.xlim([1, 2146])
plt.subplot(4, 2, 5)
plt.title('Reconstruction least square')
plt.imshow((X @ xL2).reshape(m, n), cmap='gray')
plt.subplot(4, 2, 6)
plt.title('Reconstruction L_1')
plt.imshow((X @ xL1).reshape(m, n), cmap='gray')
plt.subplot(4, 2, 7)
plt.title('Error least square')
plt.imshow((testfaces[:, num] - X @ xL2).reshape(m, n), cmap='gray')
plt.subplot(4, 2, 8)
plt.title('Sparse error L_1')
plt.imshow((testfaces[:, num] - X @ xL1).reshape(m, n), cmap='gray')
plt.show()

# Identification
num = 35
xL1 = cvx.Variable(PSI.shape[1])
objective = cvx.Minimize(cvx.norm(xL1, 1))
constraints = [cvx.norm(PSI @ xL1 - b[:, num], 2) <= .01]
prob = cvx.Problem(objective=objective, constraints=constraints)
prob.solve()
xL1 = xL1.value

Ident = np.empty(37)
for i in range(37):
    Ident[i] = np.sum(np.abs(xL1[i * 58:(i + 1) * 58]) / np.sum(np.abs(xL1)))

plt.figure(figsize=(20, 20))
plt.subplot(2, 3, 1)
plt.imshow(testfaces[:, num].reshape(m, n), cmap='gray')
plt.title(['Test Picture of Person: ', str(round(num / 2))])
plt.subplot(2, 3, 2)
plt.imshow(b[:, num].reshape(o, p), cmap='gray')
plt.title('Downsampled Picture')
plt.subplot(2, 3, 3)
plt.title('x')
plt.plot(xL1)
plt.subplot(2, 3, 4)
plt.title('Reconstruction L_1')
plt.imshow((X @ xL1).reshape(m, n), cmap='gray')
plt.subplot(2, 3, 5)
plt.title('Sparse error')
plt.imshow((testfaces[:, num] - X @ xL1).reshape(m, n), cmap='gray')
plt.subplot(2, 3, 6)
plt.bar(range(len(Ident)), Ident)
ind = np.argmax(Ident)
plt.xlabel(['Recognised Peron: ', str(ind)])
plt.show()

# Corrupt image Classification
num = 13
face = testfaces[:, num]
face = face.reshape(m, n)
glasses = imread('glasses.jpg')
rgb_weights = [.2989, .5870, .1140]
glasses = np.dot(glasses[..., :3], rgb_weights)
glasses = Image.fromarray(glasses)
size = 168, 60
glasses.thumbnail(size, Image.ANTIALIAS)
glasses = np.array(glasses)
glasses /= 255
glasses[glasses > .8] = 1

face[30:90, :] = face[30:90, :] * glasses

f = Image.fromarray(face)
size = p, o
f.thumbnail(size, Image.ANTIALIAS)
f = np.array(f)
f = f.reshape(o * p)

xL1 = cvx.Variable(PSI.shape[1])
objective = cvx.Minimize(cvx.norm(xL1, 1))
constraints = [cvx.norm(PSI @ xL1 - f, 2) <= .01]
prob = cvx.Problem(objective=objective, constraints=constraints)
prob.solve()
xL1 = xL1.value

plt.figure(figsize=(20, 20))
plt.subplot(2, 3, 1)
plt.imshow(face, cmap='gray')
plt.title(['Test Picture of Person: ', str(round(num / 2))])
plt.subplot(2, 3, 2)
plt.imshow(f.reshape(o, p), cmap='gray')
plt.title('Downsampled Picture')
plt.subplot(2, 3, 3)
plt.title('x')
plt.plot(xL1)
plt.subplot(2, 3, 4)
plt.title('Reconstruction L_1')
plt.imshow((X @ xL1).reshape(m, n), cmap='gray')
plt.subplot(2, 3, 5)
plt.title('Sparse error')
plt.imshow((face.reshape(n * m) - X @ xL1).reshape(m, n), cmap='gray')
plt.subplot(2, 3, 6)
plt.bar(range(len(Ident)), Ident)
ind = np.argmax(Ident)
plt.xlabel(['Recognised Peron: ', str(ind)])
plt.show()

# Self developed iterative solution
xopt = np.linalg.pinv(PSI) @ f
i = 0
while np.sum(abs(xopt)) > 2:
    i = i + 1
    ind = np.argsort(np.abs(xopt))
    x_s = xopt[ind]
    x_s[:i * 2] = 0
    PSI_s = PSI[:, ind]
    PSI_red = PSI_s[:, i * 2:]
    x_part = np.linalg.pinv(PSI_red) @ f
    x_s[i * 2:] = x_part
    xopt[ind] = x_s
    if np.sum(np.abs(f - PSI @ xopt)) / (o * p) > 5:
        break

for i in range(37):
    Ident[i] = np.sum(np.abs(xopt[i * 58:(i + 1) * 58]) / np.sum(np.abs(xopt)))

plt.figure(figsize=(20, 20))
plt.subplot(2, 3, 1)
plt.imshow(face, cmap='gray')
plt.title(['Test Picture of Person: ', str(round(num / 2))])
plt.subplot(2, 3, 2)
plt.imshow(f.reshape(o, p), cmap='gray')
plt.title('Downsampled Picture')
plt.subplot(2, 3, 3)
plt.title('x (optimised Least Square)')
plt.plot(xopt)
plt.subplot(2, 3, 4)
plt.title('Reconstruction')
plt.imshow((X @ xopt).reshape(m, n), cmap='gray')
plt.subplot(2, 3, 5)
plt.title('Error')
plt.imshow((face.reshape(m * n) - X @ xopt).reshape(m, n), cmap='gray')
plt.subplot(2, 3, 6)
plt.bar(range(len(Ident)), Ident)
ind = np.argmax(Ident)
plt.xlabel(['Recognised Peron: ', str(ind)])
plt.show()

# Visualisation
plt.figure(figsize=(10, 30))
plt.subplot(2, 3, 1)
f = f.reshape(-1, 1)
plt.imshow(f, cmap='gray')
plt.title('Reshaped and downsized')
plt.axis('off')
plt.subplot(2, 3, 2)
plt.imshow(PSI, cmap='gray')
plt.title('PSI')
plt.subplot(2, 3, 3)
xL1 = xL1.reshape(-1, 1)
plt.imshow(xL1, cmap='gray')
plt.axis('off')
plt.title('Sparse x')
plt.subplot(2, 3, 4)
plt.title('Downsized Image')
plt.imshow(f.reshape(o, p), cmap='gray')
plt.subplot(2, 3, 5)
plt.title('Testimage')
plt.imshow(face, cmap='gray')
plt.show()

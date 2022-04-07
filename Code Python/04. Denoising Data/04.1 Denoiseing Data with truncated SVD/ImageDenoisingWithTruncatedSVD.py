from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

# Low rank system
Image = imread('Image.jpg')
Image = np.double(Image)

image = np.mean(Image, -1)

noise = np.random.randn(455, 728)*100

noisyimage = image + noise

plt.figure(figsize=(20, 10))
c = [0, 0.4470, 0.7410]
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original image')

plt.subplot(2,3,2)
plt.imshow(noise, cmap='gray')
plt.title('noise')

plt.subplot(2,3,3)
plt.imshow(noisyimage, cmap='gray')
plt.title('noisy image')

[Uo, So, Vo] = np.linalg.svd(image, full_matrices=False)
[Un, Sn, Vn] = np.linalg.svd(noise, full_matrices=False)
[Uni, Sni, Vni] = np.linalg.svd(noisyimage, full_matrices=False)

plt.subplot(2, 3, 4)
plt.semilogy(So, color=c, linestyle='-', marker='o')
plt.title('Singular values original image')
plt.ylim([1e1, 1e5])
plt.xlim([0, len(So)])

plt.subplot(2,3,5)
plt.semilogy(Sn, color=c, linestyle='-', marker='o')
plt.title('Singular values noise')
plt.ylim([1e1, 1e5])
plt.xlim([0, len(So)])

plt.subplot(2,3,6)
plt.semilogy(Sni, color=c, linestyle='-', marker='o')
plt.title('Singular values noisy image')
plt.ylim([1e1, 1e5])
plt.xlim([0, len(So)])

plt.show()

# Computing optimal truncation

svalues = Sni
svalueslog = np.log(svalues)
med = np.ma.median(svalueslog)
indx = int(np.where(svalueslog == med)[0])

# Calculate slope
x1 = np.round(indx/2)
x2 = np.round((len(svalues)-indx)/2)

m = (svalueslog[int(indx-x1)]-svalueslog[int(indx+x2)])/(x1+x2)

# Calculate truncation value
tau = np.ma.exp(m*indx+svalueslog[indx])

plt.figure(figsize=(10, 7))
plt.semilogy(Sni, color=c, linestyle='-', marker='o')
# Noise approx line
x = np.arange(-indx, len(svalues)-indx)
plt.semilogy(np.arange(0, len(svalues)), np.transpose(np.ma.exp(-m*x+svalueslog[indx])), color='r', linewidth=2)
plt.xlim([1, len(svalues)])
plt.semilogy(indx, svalues[indx], color='b', marker='o', linewidth=4)
plt.semilogy(indx-x1, svalues[int(indx-x1)], color='c', marker='o', linewidth=4)
plt.semilogy(indx+x2, svalues[int(indx+x2)], color='m', marker='o', linewidth=4)
plt.title('Singular values and linear noise approximation')
plt.legend(['singular values', 'linear noise approximation', 'median value Theta', 'theta / 2', 'theta + (m-theta)/2'])

plt.show()

# truncation
ind = np.diag(Sni) > tau
truncatedSni = np.dot(np.diag(Sni), ind)

imagedenoised = np.dot(Uni, np.dot(truncatedSni, Vni))

plt.figure(figsize=(20, 20))
plt.subplot(2,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')

plt.subplot(2,2,2)
plt.imshow(noisyimage, cmap='gray')
plt.title('Noisy image')

plt.subplot(2,2,3)
plt.imshow(imagedenoised, cmap='gray')
plt.title('Denoised image with computet trunaction of #sv = ' + str(np.linalg.matrix_rank(truncatedSni)))


# Optimal truncation
error = np.zeros(60)

for i in range(0, 60):
    sv = Sni
    sv[59-i:len(sv)] = 0
    imagedenoised = np.dot(Uni, np.dot(np.diag(sv), Vni))
    error[i] = np.sum(np.sum(abs(imagedenoised - image), axis=0)) / (len(image[0, :])*len(image[:,0]))

error = error[::-1]
plt.subplot(2, 2, 4)
plt.plot(np.arange(0, 60), error, color=c, linestyle='-', marker='o')
minn = np.amin(error)
ind = np.where(error == minn)[0]
plt.plot(ind, minn, color='r', linewidth=4, marker='o')
plt.ylim([0, 80])
plt.xlabel('Number of kept singular values')
plt.ylabel('Weighted error (denoised-image)/#pixels')
plt.title('Error of truncated image to original')
plt.show()

#========================================================================================================

# High rank system
Image = imread('Einstein.jpg')
image = np.double(Image)


noise = np.random.randn(3668, 2945)*100

noisyimage = image + noise

[Uni, Sni, Vni] = np.linalg.svd(noisyimage, full_matrices=False)

# Computing optimal truncation

svalues = Sni
svalueslog = np.log(svalues)
med = np.ma.median(svalueslog)
indx = int(np.where(svalueslog == med)[0])

# Calculate slope
x1 = np.round(indx/2)
x2 = np.round((len(svalues)-indx)/2)

m = (svalueslog[int(indx-x1)]-svalueslog[int(indx+x2)])/(x1+x2)

# Calculate truncation value
tau = np.ma.exp(m*indx+svalueslog[indx])

# truncation
ind = np.diag(Sni) > tau
truncatedSni = np.dot(np.diag(Sni), ind)

imagedenoised = np.dot(Uni, np.dot(truncatedSni, Vni))

plt.figure(figsize=(20, 20))
plt.subplot(2,2,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')

plt.subplot(2,2,2)
plt.imshow(noisyimage, cmap='gray')
plt.title('Noisy image')

plt.subplot(2,2,3)
plt.imshow(imagedenoised, cmap='gray')
plt.title('Denoised image with computet trunaction of #sv = ' + str(np.linalg.matrix_rank(truncatedSni)))


# Optimal truncation
error = np.zeros(60)

for i in range(0, 60):
    sv = Sni
    sv[59-i:len(sv)] = 0
    imagedenoised = np.dot(Uni, np.dot(np.diag(sv), Vni))
    error[i] = np.sum(np.sum(abs(imagedenoised - image), axis=0)) / (len(image[0, :])*len(image[:,0]))

error = error[::-1]
plt.subplot(2, 2, 4)
plt.plot(np.arange(0, 60), error, color=c, linestyle='-', marker='o')
minn = np.amin(error)
ind = np.where(error == minn)[0]
plt.plot(ind, minn, color='r', linewidth=4, marker='o')
plt.ylim([0, 80])
plt.xlabel('Number of kept singular values')
plt.ylabel('Weighted error (denoised-image)/#pixels')
plt.title('Error of truncated image to original')
plt.show()
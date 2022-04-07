import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import imread

## Import Data White Noise
imageimp = imread('Einstein.jpg')
image = np.double(imageimp)

noise = np.random.randn(3668, 2945)*100

noisyimage = image + noise

## Compute Fourier Coefficient matrix

FC = np.fft.fft2(noisyimage)

## Filter Noise Frequencies
# only pass low frequencies (middle)

FCshifted = np.fft.fftshift(FC)
m = round(3668/2)
n = round(2945/2)

a = 190
FCclean = np.zeros((3668, 2945), dtype=complex)
FCclean[m-a:m+a, n-a:n+a] = FCshifted[m-a:m+a, n-a:n+a]


## Reconstruct denoised image

FCclean = np.fft.fftshift(FCclean)
Imagedenoised = np.fft.ifft2(FCclean)

plt.figure(figsize=(20, 10))
plt.subplot(2,3,1)
plt.axis('off')
plt.imshow(noisyimage, cmap='gray')
plt.title('Noisy image')
plt.subplot(2,3,2)
plt.axis('off')
plt.imshow(abs(Imagedenoised), cmap='gray')
plt.title('Denoised image')
plt.axis('off')
plt.subplot(2,3,3)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.axis('off')

FClog = np.log(abs(np.fft.fftshift(FC))+1)
plt.subplot(2,3,4)
plt.imshow(FClog, cmap='gray')
plt.axis('off')

FCcleanlog = np.log(abs(np.fft.fftshift(FCclean))+1)
plt.subplot(2,3,5)
plt.imshow(FCcleanlog, cmap='gray')
plt.axis('off')

FCcleanlog = np.log(abs(np.fft.fftshift(np.fft.fft2(image)))+1)
plt.subplot(2,3,6)
plt.imshow(FCcleanlog, cmap='gray')
plt.axis('off')
plt.show()

## Import Data High Frequent Noise

Import = imread('NoisyMoonLanding.png')
Image = np.double(Import)

## Plot Noisy Picture

plt.figure()
plt.imshow(Image, cmap='gray')
plt.axis('off')
plt.show()

## Compute Fourier Coefficient matrix

FC = np.fft.fft2(Image)

## Filter Noise Frequencies
# Noise Frequnences should be the dominant frequences of the picture.

FClog = np.log(abs(np.fft.fftshift(FC))+1)
plt.figure()
plt.subplot(2,1,1)
plt.imshow(FClog, cmap='gray')
plt.axis('off')

# only pass low frequencies (middle)

FCshifted = np.fft.fftshift(FC)
m = round(474/2)
n = round(630/2)

a = 44
FCclean = np.zeros((474, 630), dtype=complex)
FCclean[m-a:m+a, n-a:n+a] = FCshifted[m-a:m+a, n-a:n+a]

FCcleanlog = np.log(abs(FCclean)+1)
plt.subplot(2,1,2)
plt.imshow(FCcleanlog, cmap='gray')
plt.axis('off')
plt.show()

## Reconstruct denoised image

FCclean = np.fft.fftshift(FCclean)
Imagedenoised = np.fft.ifft2(FCclean)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(Image, cmap='gray')
plt.title('Noisy image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(abs(Imagedenoised), cmap='gray')
plt.title('Denoised image')
plt.axis('off')
plt.show()

## Filtering Noise special Frequencies

FClog = np.log(abs(np.fft.fftshift(FC))+1)
X, Y = np.meshgrid(np.arange(1, np.shape(FClog)[1]+1), np.arange(1, np.shape(FClog)[0]+1))
figs = plt.figure(figsize=(10, 10))
sp = figs.add_subplot(211, projection='3d')
mappable = sp.plot_surface(X, Y, FClog, cmap='jet')

ind = FClog < 3.9
a = 44
b = 110
indinner = FClog < 4.8
FCclean2 = FCshifted*ind
FCclean2inner = FCshifted*indinner
FCclean2[m-b:m+b, n-b:n+b] = FCclean2inner[m-b:m+b, n-b:n+b]
FCclean2[m-a:m+a, n-a:n+a] = FCshifted[m-a:m+a, n-a:n+a]

FCclean2log = np.log(abs(FCclean2)+1)
plt.subplot(2,1,2)
plt.imshow(FCclean2log, cmap='gray')
plt.axis('off')
plt.show()


FCclean = np.fft.fftshift(FCclean2)
Imagedenoised = np.fft.ifft2(FCclean2)

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
plt.imshow(Image, cmap='gray')
plt.axis('off')
plt.title('Noisy image')
plt.subplot(1,2,2)
plt.imshow(abs(Imagedenoised),cmap='gray')
plt.axis('off')
plt.title('Denoised image')
plt.show()

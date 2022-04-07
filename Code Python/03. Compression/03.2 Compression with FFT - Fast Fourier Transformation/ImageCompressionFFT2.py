from scipy.linalg import dft
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

Image = imread('03. Compression/03.2 Compression with FFT - Fast Fourier Transformation/Einstein.jpg')
Image = np.double(Image)

# Surface plot
columnindices = np.arange(0, 2945)
columnindices[::-1].sort()
X, Y = np.meshgrid(np.arange(1, np.shape(Image)[1]+1), np.arange(1, np.shape(Image)[0]+1))
fig0 = plt.figure(figsize=(10, 10))
sp = fig0.add_subplot(111, projection='3d')
mappable = sp.plot_surface(X, Y, Image[:, columnindices], cmap='jet')
sp.view_init(65, 100)
fig0.colorbar(mappable)
plt.show()

# Plotting Discrete Fourier Transformation Matrix
F = dft(3668)
fig1 = plt.figure(figsize=(20, 20))
plt.imshow(np.real(F), cmap='gray')
plt.title('DFT matrix')

# Fourier Transormation
FourImage = np.fft.ifft2(Image)

# Reducing Data
# Percentages to which we reduce our data
perc = [0.99, 0.05, 0.01, 0.001]

# Fourier Coefficients sorted for finding % of largest
FourOrder = np.abs(FourImage.flatten())
FourOrder[::-1].sort()

fig2 = plt.figure(figsize=(20, 10))

# Looping percentages
for i in range(0, len(perc)):
    limit = FourOrder[int(np.round(perc[i]*len(FourOrder)))]
    ind = abs(FourImage) > limit
    RedFourier = FourImage*ind
    RedImage = np.rot90(np.rot90(np.fft.ifft2(RedFourier)))
    plt.subplot(2, 4, i+1)
    plt.imshow(np.abs(RedImage), cmap='gray')
    plt.axis('off')
    plt.title('Coefficients reduced to ' + str(perc[i-1]*100) + '%')
    plt.subplot(2, 4, i+5)
    Frame = RedImage[1000:1500, 1000:1700]
    plt.imshow(np.abs(Frame), cmap='gray')
    plt.axis('off')
    plt.title('Coefficients reduced to ' + str(perc[i-1]*100) + '%')

plt.show()

# Edge Detection
modes = [50, 500, 1000, 2000]
fig3 = plt.figure(figsize=(20, 10))

for i in range(0, len(modes)):
    limit = FourOrder[modes[i]]
    ind = abs(FourImage) < limit
    RedFourier = FourImage*ind
    RedImage = np.rot90(np.rot90(np.fft.ifft2(RedFourier)))
    plt.subplot(2, 4, i+1)
    plt.imshow(np.abs(RedImage), cmap='gray')
    plt.axis('off')
    plt.title('Coefficients reduced by ' + str(modes[i]) + ' frequencies')
    plt.subplot(2, 4, i+5)
    RedFourierlog = np.log(np.abs(np.fft.fftshift(RedFourier))+1)
    plt.imshow(np.abs(RedFourierlog), cmap='gray')
    plt.axis('off')
    plt.title('Coefficients reduced by ' + str(modes[i]) + ' frequencies')

plt.show()

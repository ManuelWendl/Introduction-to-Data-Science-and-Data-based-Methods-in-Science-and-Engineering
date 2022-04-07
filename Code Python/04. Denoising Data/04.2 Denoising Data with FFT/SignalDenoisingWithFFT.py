import matplotlib.pyplot as plt
import numpy as np

dt = 0.0005
t = np.arange(0, 1, dt)

audio = 4*np.sin(2*np.pi*100*t) + 8*np.sin(2*np.pi*60*t) + 4*np.cos(2*np.pi*20*t) \
        + 6*np.cos(2*np.pi*5*t)
noisy = audio + 10*np.random.randn(len(t))

## Plot Audio signals

plt.figure()
plt.subplot(2,1,1)
plt.plot(t,noisy,color='r')
plt.plot(t,audio, color='k', linewidth=2)
plt.legend(['Noisy Audio','Original Audio'])

## Fast Fourier Transformation

# FC are the Fourier Coeficients
FC = np.fft.fft(noisy,len(t))

# Power of different frequencies
P = np.abs(FC)/len(t)

freq = 1/(dt*len(t))*(np.arange(0,len(t)))

plt.subplot(2,1,2)
plt.plot(freq, P)
plt.plot(freq, np.ones(len(freq))*1.5, color='r')
plt.xlim([1, 300])
plt.xlabel('frequencies in Hz')
plt.ylabel('Power of frequencies')
plt.title('Powerspectrum')
plt.legend(['Powerspectrum', 'Truncation level'])
plt.show()
## Filter Noisy set

ind = P > 1.5
FCclean = FC*ind

audiodenoised = np.fft.ifft(FCclean)

# Plot denoised and original for comparison
plt.figure()
plt.plot(t,np.real(audiodenoised), color='r', linewidth=2)
plt.plot(t,audio, color='k', linewidth=2)
plt.legend(['Denoised', 'Original'])
plt.title('Overlayed original and denoised audio Signal')
plt.show()

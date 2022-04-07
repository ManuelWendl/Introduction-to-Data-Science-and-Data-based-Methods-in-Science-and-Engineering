import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sounddevice as sd
import time
import soundfile as sf

audio, FS = sf.read('Für-Elise.wav')
audio = audio[0:300000, 0]

FC = np.fft.fft(audio)

perc = [0.99, 0.3, 0.1, 0.01, 0.001]

FCsort = abs(FC)
FCsort[::-1].sort()

plt.figure(figsize=(20, 10))

for i in range(0, len(perc)):
    limit = FCsort[int(np.round(perc[i] * len(FCsort)))]
    ind = abs(FC) > limit
    FCred = FC * ind
    audiocomp = np.fft.ifft(FCred)

    sd.play(np.real(audiocomp), FS)

    plt.subplot(2, 5, i+1)
    plt.plot(np.arange(0, len(audio)), np.real(audiocomp))
    plt.axis([1, len(audio), -1, 1])
    plt.title('compressed to ' + str(perc[i] * 100) + ' %')

    plt.subplot(2, 5, i + 6)
    PS = abs(FCred)
    plt.semilogy(np.arange(0, len(audio)), np.abs(np.fft.fftshift(PS)))
    plt.ylim([1e-3, 1e+4])
    time.sleep(len(audio)/FS + .5)
plt.show()

plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.imshow(imread('NotesFür-Elise.jpg'))
plt.title('Tones')

dF = FS / len(audio)
f = np.arange(-FS/2, FS/2, dF)

# Defining colors
c = [0, 0.4470, 0.7410]
r = [1, 0, 0]
b = [0, 0, 1]
g = [0, 1, 0]
y = [1, 1, 0]

plt.subplot(212)
plt.axis([0, 3000, 0, 2200])
plt.plot(329.427, 2099.4, color=r, marker='o', linewidth=3)
plt.plot(439.677, 1119.72, color=b, marker='o', linewidth=3)
plt.plot(494.214, 1116.68, color=g, marker='o', linewidth=3)
plt.plot(621.81, 1091.37, color=y, marker='o', linewidth=3)
plt.plot(f, abs(np.fft.fftshift(FC)), color=c)
plt.plot(np.ones(2200)*329.6, np.arange(1, 2201), color='k')
plt.plot(np.ones(2200)*392, np.arange(1, 2201), color='k')
plt.plot(np.ones(2200)*493.8, np.arange(1, 2201), color='k')
plt.plot(np.ones(2200)*587.3, np.arange(1, 2201), color='k')
plt.plot(np.ones(2200)*698.5, np.arange(1, 2201), color='k')
plt.xlabel('frequencies in Hz')
plt.ylabel('|f| Power of frequency')
plt.title('Power Spectrum')
plt.legend(['E4', 'A4', 'B4', 'Dis5', 'Power Spectrum'])
plt.show()

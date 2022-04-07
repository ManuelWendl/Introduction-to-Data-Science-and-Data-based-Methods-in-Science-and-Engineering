import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

# Create Signal

dt = .001
t = np.arange(0, 1, dt)

f = np.sin(2*np.pi*10*t)+np.sin(2*np.pi*70*t) + np.sin(2*np.pi*160*t) + np.cos(2*np.pi*120*t)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, f, linewidth=2)
plt.title('Original signal')

FC = np.fft.fft(f)
P = np.abs(FC)/len(t)

freq = 1/(dt * len(t))*(np.arange(1, len(t)+1, 1))
plt.subplot(2, 1, 2)
plt.plot(freq, P, linewidth=2)
plt.xlim([1, 500])
plt.xlabel('frequencies in Hz')
plt.ylabel('Power of frequencies')
plt.title('Powerspectrum')
plt.show()

# Downsample Data
s = 55
red = np.random.randint(1000, size=s)
X = f[red]

plt.figure()
plt.plot(t,f,linewidth=2)
plt.plot(red/1001,X,'xr',linewidth=4)
plt.legend(['Original signal','Random samples'])
plt.title('Measurement samples')
plt.show()

# Create PSI and determine L1 minimization
DFT = np.conj(np.fft.fft(np.eye(1000))/1000)
PSI = DFT[red, :]

xL1 = cvx.Variable(1000, complex='True')
obj = cvx.Minimize(cvx.norm(xL1,1))
constraints = [cvx.sum_squares(PSI@xL1-X) <= 1]
prob = cvx.Problem(obj, constraints)
prob.solve()

xL1 = xL1.value
print(xL1)
xL2 = np.linalg.pinv(PSI) @ X

sigl1 = np.fft.ifft(xL1)
sigl2 = np.fft.ifft(xL2)

Pl1 = np.abs(xL1)/len(t)
Pl2 = np.abs(xL2)/len(t)

plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
plt.title('Compressed Sensing with L1 norm')
plt.plot(t,f,linewidth=2)
plt.plot(t,np.real(sigl1),'r',linewidth=2)
plt.plot(red/1001,X,'xk',linewidth=2)
plt.legend(['Original signal','Reconstructed signal','Measurement samples'])
plt.subplot(2,2,3)
plt.plot(freq,Pl1,linewidth=2)
plt.xlabel('frequencies in Hz')
plt.ylabel('Power of frequencies')
plt.xlim([1, 500])
plt.title('Powerspectrum')

plt.subplot(2,2,2)
plt.title('Least square approximation')
plt.plot(t,f,linewidth=2)
plt.plot(t,np.real(sigl2),'r',linewidth=2)
plt.plot(red/1001,X,'xk',linewidth=2)
plt.legend(['Original signal','Reconstructed signal','Measurement samples'])
plt.subplot(2,2,4)
plt.plot(freq,Pl2,linewidth=2)
plt.xlabel('frequencies in Hz')
plt.ylabel('Power of frequencies')
plt.xlim([1, 500])
plt.title('Powerspectrum')

plt.show()

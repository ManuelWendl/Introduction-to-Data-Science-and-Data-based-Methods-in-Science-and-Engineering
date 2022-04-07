import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

# Nyquist and Shannon Sampling Theorem

dt = .001
t = np.arange(0, 1, dt)

f = np.sin(2*np.pi*10*t)

smplrate = np.arange(1, 1001, 20)
samples = f[smplrate]

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.title('Sufficiently sampled signal')
plt.plot(f)
plt.plot(smplrate, samples, 'xr')
plt.xlim([1, 1000])
plt.legend(['Original and reconstructed signal', 'Samples'], loc='upper right')
plt.show()

smplrate = np.arange(1, 1001, 90)
samples = f[smplrate]

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.title('Undersampled Signal')
plt.plot(f)
plt.plot(smplrate, samples, 'xr')
plt.plot(-np.sin(2*np.pi*1.09*t))
plt.xlim([1, 1000])
plt.legend(['Original and reconstructed signal', 'Samples','Reconstructed Signal'], loc='upper right')
plt.show()

# Compressed Sensing

dt = .01
t = np.arange(0, 1, dt)

f = np.sin(2*np.pi*2*t) + np.cos(2*np.pi*5*t)

smplrate = np.random.randint(100, size=(30, 1))
samples = f[smplrate]

plt.figure(figsize=(10, 5))
plt.title('Randomly Sampled')
plt.plot(f)
plt.plot(smplrate, samples, 'xr')
plt.xlim([1, 100])
plt.legend(['Original signal', 'Samples'])
plt.show()

Orth = np.conj(np.fft.fft(np.eye(101))/101)
S = np.zeros((len(smplrate), 101))

for i in range(1, len(samples)):
    S[i, smplrate[i]] = 1

samples = samples.reshape(len(samples), 1)

plt.figure(figsize=(10, 20))
plt.subplot(2,5,1)
plt.imshow(samples, cmap='gray')
plt.axis('off')
plt.title('Sampled Data')
plt.subplot(2,5,3)
plt.imshow(np.real(Orth), cmap='gray')
plt.axis('off')
plt.title('Phi')
plt.subplot(2,5,2)
plt.imshow(S, cmap='gray')
plt.axis('off')
plt.title('R')

samples = samples.reshape(len(samples))

PSI = S @ Orth

xL1 = cvx.Variable(101, complex='True')
obj = cvx.Minimize(cvx.norm(xL1, 1))
constraints = [cvx.sum_squares(PSI@xL1-samples) <= 1]
prob = cvx.Problem(obj, constraints)
prob.solve()
x = xL1.value
print(x)

x = x.reshape(len(x), 1)
samples = samples.reshape(len(samples), 1)

plt.subplot(2, 5, 4)
plt.imshow(np.abs(x), cmap='gray')
plt.axis('off')
plt.title('Sparse L_1 x')
plt.subplot(2, 5, 5)
plt.imshow(abs(np.linalg.pinv(S.dot(Orth)).dot(samples)), cmap='gray')
plt.axis('off')
plt.title('Not sparse L_2 x')
plt.subplot(2, 5, 7)
plt.imshow(np.real(np.dot(S, Orth)), cmap='gray')
plt.axis('off')
plt.title('Psi')
plt.show()


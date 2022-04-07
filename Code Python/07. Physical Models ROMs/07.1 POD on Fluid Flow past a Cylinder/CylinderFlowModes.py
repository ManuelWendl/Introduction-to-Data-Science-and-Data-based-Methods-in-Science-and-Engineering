from PIL import Image, ImageSequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cvx

# Import Data
gif = Image.open("CylinderFlow.gif")

X = np.empty((38700, 226))
k = 0
for frame in ImageSequence.Iterator(gif):
    x = np.array(frame)
    x = x[90:180, 50:]
    X[:, k] = x.reshape(-1)
    k += 1

fig = plt.figure()
gifup = []
for i in range(gif.n_frames):
    gifup.append([plt.imshow(X[:, i].reshape(90, 430), animated=True, cmap='gray')])

ani = animation.ArtistAnimation(fig, gifup, interval=260, blit=True, repeat_delay=100)
ani.save("Flow.gif")

# Principal Modes
# PCA of timeframes

u, s, v = np.linalg.svd(X, full_matrices=False)

plt.figure()
plt.subplot(1, 2, 1)
plt.title('Singular values')
plt.semilogy(s[:50], 'o-')
plt.subplot(1, 2, 2)
svar = np.empty(s.shape)
for i in range(len(s)):
    svar[i] = sum(s[0:i]) / sum(s)
plt.plot(svar)
plt.title('Captured variance')
plt.show()

# PLot first POD Modes
plt.figure(figsize=(10, 20))
plt.subplot(5, 2, 1)
plt.imshow(np.abs(u[:, 0].reshape(90, 430)), cmap='jet')
plt.title('Mean Flow')
for i in range(3, 11):
    plt.subplot(5, 2, i)
    plt.imshow(np.abs(u[:, i - 2].reshape(90, 430)), cmap='jet')
    plt.title('POD' + str(i - 1))
plt.show()

# Time Correlation

plt.figure(figsize=(10, 20))
for i in range(1, 19, 2):
    plt.subplot(9, 2, i)
    plt.imshow(np.abs(u[:, i - 1].reshape(90, 430)), cmap='jet')
    plt.subplot(9, 2, i + 1)
    plt.plot(v[i - 1, :])
plt.show()

# Compressed Sensing with time samples

R = np.round(np.linspace(0, 216, 40))
R = R + np.random.randint(5, size=40)

Xr = X[:, np.int_(R)]

ur, sr, vr = np.linalg.svd(Xr, full_matrices=False)

plt.figure(figsize=(10, 20))
plt.subplot(5, 2, 1)
plt.imshow(np.abs(ur[:, 0].reshape(90, 430)), cmap='jet')
plt.title('Mean Flow')
for i in range(3, 11):
    plt.subplot(5, 2, i)
    plt.imshow(np.abs(ur[:, i - 2].reshape(90, 430)), cmap='jet')
    plt.title('POD' + str(i - 1))
plt.show()

plt.figure(figsize=(10, 20))
for i in range(1, 19, 2):
    plt.subplot(9, 2, i)
    plt.imshow(np.abs(ur[:, i - 1].reshape(90, 430)), cmap='jet')
    plt.subplot(9, 2, i + 1)
    plt.plot(vr[i - 1, :])
plt.show()

PHI = np.fft.fft(np.eye(226))
PSI = PHI[np.int_(R),:]

vrec = np.empty((v.shape[1], 30))

for i in range(30):
    b = vr[i, :]
    xL1 = cvx.Variable(PSI.shape[1], complex=True)
    objective = cvx.Minimize(cvx.norm(xL1, 1))
    constraints = [cvx.norm(PSI @ xL1 - b, 2) <= .05]
    prob = cvx.Problem(objective, constraints)
    result = prob.solve()
    vrec[:, i] = np.real(np.fft.ifft(xL1.value))

plt.figure(figsize=(10, 20))
for i in range(1, 19, 2):
    plt.subplot(9, 2, i)
    plt.imshow(np.abs(ur[:, i - 1].reshape(90, 430)), cmap='jet')
    plt.subplot(9, 2, i + 1)
    plt.plot(vrec[:, i-1])
plt.show()


# Reconstruct Flow
vrec_f = np.empty(vrec.shape)

f = np.fft.fftshift(np.fft.fft(vrec[:,0]))
f[1:80] = 0
f[150:] = 0
vrec_f[:, 0] = np.real(np.fft.ifft(np.fft.fftshift(f)))

for i in range(1, 30):
    f = np.fft.fftshift(np.fft.fft(vrec[:, i]))
    f[1:85] = 0
    f[135:] = 0
    vrec_f[:, i] = np.real(np.fft.ifft(np.fft.fftshift(f)))

plt.figure(figsize=(10, 20))
for i in range(1, 19, 2):
    plt.subplot(9, 2, i)
    plt.imshow(np.abs(ur[:, i - 1].reshape(90, 430)), cmap='jet')
    plt.subplot(9, 2, i + 1)
    plt.plot(vrec_f[:, i-1])
plt.show()

Rec = (ur[:,0:30].dot(np.diag(sr)[0:30,0:30])).dot(vrec_f[:,0:30].T)

fig = plt.figure()
gifup = []
for i in reversed(list(range(gif.n_frames))):
    gifup.append([plt.imshow(Rec[:, i].reshape(90, 430), animated=True, cmap='gray')])

ani = animation.ArtistAnimation(fig, gifup, interval=260, blit=True, repeat_delay=100)
ani.save("Rec.gif")



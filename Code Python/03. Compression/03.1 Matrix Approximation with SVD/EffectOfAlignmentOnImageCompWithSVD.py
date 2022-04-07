import numpy as np
import matplotlib.pyplot as plt

M = np.zeros((201, 201))
M0 = M
M0[90: 110, :] = 1

fig1 = plt.figure(figsize=(20, 10))
plt.subplot(2, 6, 1)
plt.imshow(M0, cmap='gray')
plt.axis('off')

[U, S, V] = np.linalg.svd(M0, full_matrices=False)
plt.subplot(2, 6, 7)
plt.plot(S[0:29], linewidth=2, color='k', linestyle='-', marker='o')

x = range(1, 202)
for i in range(1, 5):
    f = np.int8(np.round(i * 10 * np.sin(np.dot(2 * np.pi / 201, x))))
    Mi = np.zeros((201, 201))
    for ii in range(1, 201):
        Mi[(90 + f[ii]):(110 + f[ii]), ii] = 1

    plt.subplot(2, 6, i+1)
    plt.imshow(Mi, cmap='gray')
    plt.axis('off')

    [U, S, V] = np.linalg.svd(Mi, full_matrices=False)
    plt.subplot(2, 6, i + 7)
    plt.plot(S[0: 29], linewidth=2, color='k', linestyle='-', marker='o')

plt.show()



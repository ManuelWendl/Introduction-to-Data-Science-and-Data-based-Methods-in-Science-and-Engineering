from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

Import = imread('BW_Picture.jpg')
Gray = np.mean(Import, -1)

[U, S, V] = np.linalg.svd(Gray, full_matrices=False)

rank_of_S = [10, 30, 50, 70, 120, 200, 300, 408];
fig1 = plt.figure(figsize=(10, 20))

for i in range(1, 9):
    v_1 = np.ones(408)
    v_1[rank_of_S[i-1]: 407] = 0
    D_S = np.diag(v_1)
    S_1 = np.matmul(np.diag(S), D_S)

    Matrix_Pic_reduced = reduce(np.dot, [U, S_1, V])
    plt.subplot(4, 2, i)
    plt.axis('off')
    plt.title('Rank: '+str(rank_of_S[i-1]))
    plt.imshow(Matrix_Pic_reduced, cmap='gray')
plt.show()

fig2 = plt.figure()
plt.semilogy(range(1, 200, 1), S[0:199], linewidth=2)
plt.title('First 200 Singular Values')
plt.show()


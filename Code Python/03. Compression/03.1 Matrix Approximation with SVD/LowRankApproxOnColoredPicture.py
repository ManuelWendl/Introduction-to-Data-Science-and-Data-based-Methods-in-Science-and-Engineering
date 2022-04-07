from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

Import = np.double(imread('Color_Picture.png'))

I_R = Import[:, :, 0]

I_G = Import[:, :, 1]

I_B = Import[:, :, 2]

[U_R, S_Ra, V_R] = np.linalg.svd(I_R, full_matrices=False)
[U_G, S_Ga, V_G] = np.linalg.svd(I_G, full_matrices=False)
[U_B, S_Ba, V_B] = np.linalg.svd(I_B, full_matrices=False)

rank_of_S = [10, 30, 50, 70, 120, 200, 300, 563]

fig1 = plt.figure(figsize=(10, 20))

for i in range(1, 9):
    v_1 = np.ones(563)
    v_1[rank_of_S[i-1]: 562] = 0
    D_S = np.diag(v_1)
    S_R = np.matmul(np.diag(S_Ra), D_S)
    S_G = np.matmul(np.diag(S_Ga), D_S)
    S_B = np.matmul(np.diag(S_Ba), D_S)

    Red_red = reduce(np.dot, [U_R, S_R, V_R])
    Green_red = reduce(np.dot, [U_G, S_G, V_G])
    Blue_red = reduce(np.dot, [U_B, S_B, V_B])

    Picture_red = np.uint8(np.dstack([Red_red, Green_red, Blue_red]))

    plt.subplot(4, 2, i)
    plt.axis('off')
    plt.title('Rank: '+str(rank_of_S[i-1]))
    plt.imshow(Picture_red)
plt.show()

fig2 = plt.figure()
plt.semilogy(range(1, 200, 1), S_Ra[0:199], linewidth=2, color='r')
plt.semilogy(range(1, 200, 1), S_Ga[0:199], linewidth=2, color='g')
plt.semilogy(range(1, 200, 1), S_Ba[0:199], linewidth=2, color='b')
plt.title('First 200 Singular Values')
plt.legend(['singular values R','singular values G','singular values B'])
plt.show()
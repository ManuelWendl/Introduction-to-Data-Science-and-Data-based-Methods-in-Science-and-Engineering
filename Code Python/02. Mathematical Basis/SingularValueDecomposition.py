import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

# Build unit sphere
a = np.linspace(-np.pi, np.pi, 100)
b = np.linspace(0, np.pi, 100)

x = np.outer(np.cos(a), np.sin(b))
y = np.outer(np.sin(a), np.sin(b))
z = np.outer(np.ones(np.size(a)), np.cos(b))

# Build linear Transformation
a1 = np.pi / 15
a2 = -np.pi / 9
a3 = -np.pi / 20

Rx = [[1, 0, 0],
      [0, np.cos(a1), -np.sin(a1)],
      [0, np.sin(a1), np.cos(a1)]]

Ry = [[np.cos(a2), 0, -np.sin(a2)],
      [0, 1, 0],
      [np.sin(a2), 0, np.cos(a2)]]

Rz = [[np.cos(a3), -np.sin(a3), 0],
      [np.sin(a3), np.cos(a3), 0],
      [0, 0, 1]]

E = np.diag([2, 1, 0.5])

A = reduce(np.dot, [Rz, Ry, Rx, E])

print(A)

# Transform sphere
xt = np.zeros(np.shape(x))
yt = np.zeros(np.shape(y))
zt = np.zeros(np.shape(z))

for i in range(1, 100):
    for ii in range(1, 100):
        v = [x[i, ii], y[i, ii], z[i, ii]]
        vt = np.matmul(A, v)
        xt[i, ii] = vt[0]
        yt[i, ii] = vt[1]
        zt[i, ii] = vt[2]

# Proess SVD
[U, S, V] = np.linalg.svd(A)

xu = np.zeros(np.shape(x))
yu = np.zeros(np.shape(y))
zu = np.zeros(np.shape(z))

for i in range(1, 100):
    for ii in range(1, 100):
        v = [x[i, ii], y[i, ii], z[i, ii]]
        vt = np.matmul(U, v)
        xu[i, ii] = vt[0]
        yu[i, ii] = vt[1]
        zu[i, ii] = vt[2]

xus = np.zeros(np.shape(x))
yus = np.zeros(np.shape(y))
zus = np.zeros(np.shape(z))

for i in range(1, 100):
    for ii in range(1, 100):
        v = [x[i, ii], y[i, ii], z[i, ii]]
        vt = reduce(np.dot, [U, np.diag(S), v])
        xus[i, ii] = vt[0]
        yus[i, ii] = vt[1]
        zus[i, ii] = vt[2]

xusv = np.zeros(np.shape(x))
yusv = np.zeros(np.shape(y))
zusv = np.zeros(np.shape(z))

for i in range(1, 100):
    for ii in range(1, 100):
        v = [x[i, ii], y[i, ii], z[i, ii]]
        A = reduce(np.dot, [U, np.diag(S), V])
        vt = np.matmul(A, v)
        xusv[i, ii] = vt[0]
        yusv[i, ii] = vt[1]
        zusv[i, ii] = vt[2]

print(A)

# Plotting
fig = plt.figure(figsize=(30, 15))
sp1 = fig.add_subplot(241, projection='3d')
sp1.plot_surface(x, y, z, cmap='jet')
sp1.title.set_text('Unit Sphere')

sp2 = fig.add_subplot(244, projection='3d')
sp2.plot_surface(xt, yt, zt, cmap='jet')
sp2.title.set_text('A * Unit Sphere')

sp3 = fig.add_subplot(245, projection='3d')
sp3.plot_surface(x, y, z, cmap='jet')
sp3.title.set_text('Unit Sphere')


sp4 = fig.add_subplot(246, projection='3d')
sp4.plot_surface(xu, yu, zu, cmap='jet')
sp4.title.set_text('U*Unit Sphere')

sp4 = fig.add_subplot(247, projection='3d')
sp4.plot_surface(xus, yus, zus, cmap='jet')
sp4.title.set_text('U*S*Unit Sphere')

sp5 = fig.add_subplot(248, projection='3d')
sp5.plot_surface(xusv, yusv, zusv, cmap='jet')
sp5.title.set_text('U*S*V*Unit Sphere')

plt.show()

import numpy as np
import matplotlib.pyplot as plt


dx = 0.001
x = np.arange(-np.pi, np.pi, dx)

f_x = 0*x
lx = len(x)
i_1 = np.arange(round(lx/6), round(lx/3))
i_2 = np.arange(round(lx/3), round(lx*5/6))

f_x[i_1] = 3*(np.arange(1, len(i_1)+1))
f_x[i_2] = -3*(np.arange(1, len(i_2)+1)) + f_x[round(lx/3)-1]

# the basis is the following formula
#   #
#   #   f(x) = ∑ a*cos(k*2π*x/L) + b*sin(k*2*π*x/L)
#   #
# For finding the coefficients we make the dotproduct of
#   #
#   #   <f(x),cos(k*2π*x/L)> and <f(x),sin(k*2π*x/L)>
#   #
# The first coefficient is gained by the integral of the area devided by two

fFc = (sum(f_x*np.ones(np.shape(x)))*dx/np.pi)/2    # First Fourier Coefficient (vertical offset)
                                                    # Fourier Series first only first fourier coefficient for vertical offset
# The Approximation will be done by 50 terms of fourier (not infinite sum)

i = 1
m = [1, 5, 10, 25, 50]

fig = plt.figure()
plt.plot(x, f_x)


FS = -fFc

for k in range(0, 49):
    a = 1/np.pi * sum(f_x * np.cos(k*x))*dx
    b = 1/np.pi * sum(f_x * np.sin(k*x))*dx
    FS = FS + a * np.cos(k*x) + b * np.sin(k*x)
    if k == m[i]:
        plt.plot(x, FS)
        i = i+1
plt.legend(['f(x)','order: 1','order: 5','order: 10','order: 25','order: 50'])
plt.show()

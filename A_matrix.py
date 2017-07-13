import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy.misc import factorial
from scipy.integrate import simps
from scipy.linalg import norm
from numpy import fft

x = np.linspace(-10., 10., 512)
omega = 3.0


def factor(n, d):
    return (1./np.sqrt(2**n * factorial(n)))*((omega/np.pi)**0.25)*np.exp(-0.5*omega*(x-d)**2)\
           *eval_hermite(n, np.sqrt(omega)*(x-d))


def overlap(d, m, n):
    if m > n:
        data = factor(m, 0.) * factor(n, displacement)
    else:
        data = np.zeros_like(x)
    result = simps(data, x)
    return result

displacement = 0.3

n_max = 4


def A_matrix(k):
    F_mk = []
    F_jn = []
    for m in range(2*n_max):
        if m in range(n_max) and k in range(n_max, 2*n_max):
            F_mk.append(overlap(displacement, m, k-n_max))
        elif m in range(n_max, 2*n_max) and k in range(n_max):
            F_mk.append(overlap(displacement, m-n_max, k))
        elif m == k:
            F_mk.append(1.0)
        else:
            F_mk.append(0.0)

    for n in range(2*n_max):
        if k-1 in range(n_max) and n in range(n_max, 2*n_max):
            F_jn.append(overlap(displacement, k-1, n-n_max))
        elif k-1 in range(n_max, 2*n_max) and n in range(n_max):
            F_jn.append(overlap(displacement, k-1-n_max, n))
        elif n == k-1:
            F_jn.append(1.0)
        else:
            F_jn.append(0.0)
    F = np.outer(np.asarray(F_mk), np.asarray(F_jn))
    F = np.abs(F)
    F = F.T

    return F

plt.figure()
for i in range(3):
    plt.plot(x, factor(i, displacement) + i*3.)
    plt.plot(x, factor(i, 1.5) + 36. + i*3.)
plt.plot(x[102: 410], 0.5*(3.*x[102: 410])**2, 'k')
plt.plot(x[186: 480], 36. + 0.5*(3.*(x[186: 480] - 1.5))**2, 'k')
plt.ylim(0.0, 90.)
# plt.show()

np.set_printoptions(precision=6)

with open("/home/ayanc/PycharmProjects/Switches/Cph8_10lvl/A_matrix.txt", "w") as f:
    for k in range(1, 2*n_max)[::-1]:
        print "Matrix A" + str(k) + str(k-1)
        print A_matrix(k)
        print "\n"

        A_matrix(k).tofile(f, sep=" ", format="%2.6lf")
        f.write(" ")

with open("/home/ayanc/PycharmProjects/Switches/Cph8_10lvl/A_matrix.txt", "r") as f:
    for line in f:
        B_mat = line.split()
        B_mat = np.array([float(x) for x in B_mat])

shp = 2*n_max*2*n_max
for k in range(1, 2*n_max)[::-1]:
    print "Matrix A" + str(k) + str(k - 1) + "read from file"
    print B_mat[(2*n_max-1 - k)*shp:(2*n_max-1 - k+1)*shp].reshape(2*n_max, 2*n_max)
    print

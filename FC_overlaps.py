import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy.misc import factorial
from scipy.integrate import simps

x = np.linspace(-4., 4., 256)
disp = np.linspace(0., 4., 200)
omega = 3.0


def factor(n, d):
    return (1./np.sqrt(2**n * factorial(n)))*((omega/np.pi)**0.25)*np.exp(-0.5*omega*(x-d)**2)\
           *eval_hermite(n, np.sqrt(omega)*x)


def overlap(disp):
    result1 = np.empty(disp.size)
    result2 = np.empty(disp.size)
    result3 = np.empty(disp.size)
    result4 = np.empty(disp.size)
    result5 = np.empty(disp.size)
    for i, d in enumerate(disp):
        data1 = factor(0, 0.)*factor(0, d)
        data2 = factor(0, 0.)*factor(1, d)
        data3 = factor(0, 0.)*factor(2, d)
        data4 = factor(0, 0.)*factor(3, d)
        data5 = factor(0, 0.)*factor(4, d)
        result1[i] = simps(data1, x)
        result2[i] = simps(data2, x)
        result3[i] = simps(data3, x)
        result4[i] = simps(data4, x)
        result5[i] = simps(data5, x)
    return result1, result2, result3, result4, result5

plt.figure()
plt.suptitle('FC overlap as a function of displacement')
plt.plot(disp, overlap(disp)[0], label='0-0 overlap')
plt.plot(disp, overlap(disp)[1], label='0-1 overlap')
plt.plot(disp, overlap(disp)[2], label='0-2 overlap')
plt.plot(disp, overlap(disp)[3], label='0-3 overlap')
plt.plot(disp, overlap(disp)[4], label='0-4 overlap')
plt.xlabel('Displacement (in a.u.)')
plt.ylabel('Overlap')
plt.legend()
plt.show()

print

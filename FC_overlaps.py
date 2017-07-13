import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from scipy.misc import factorial
from scipy.integrate import simps
from scipy.linalg import norm
from numpy import fft

x = np.linspace(-4., 4., 256)
disp = np.linspace(0., 4., 200)
# omega = 3.0


def factor(n, d, omega):
    return (1./np.sqrt(2**n * factorial(n)))*((omega/np.pi)**0.25)*np.exp(-0.5*omega*(x-d)**2)\
           *eval_hermite(n, np.sqrt(omega)*(x-d))


def overlap(d, n, m, omega):
    data = factor(m, 0., omega)*factor(n, d, omega)
    result = simps(data, x)
    return result

# plt.figure()
# plt.suptitle('FC overlap as a function of displacement')
# plt.plot(disp, [overlap(d, 0, 0) for d in disp], label='0-0 overlap')
# plt.plot(disp, [overlap(d, 0, 1) for d in disp], label='0-1 overlap')
# plt.plot(disp, [overlap(d, 0, 2) for d in disp], label='0-2 overlap')
# plt.plot(disp, [overlap(d, 0, 3) for d in disp], label='0-3 overlap')
# plt.plot(disp, [overlap(d, 0, 4) for d in disp], label='0-4 overlap')
# plt.xlabel('Displacement (in a.u.)')
# plt.ylabel('Overlap')
# plt.legend()
# plt.show()

t_max = 1765.376
t_num = 2062
dt = 2.*t_max/t_num

d_freq = np.pi / t_max
freq_max = d_freq * t_num /2.

kT = 0.1

t = np.linspace(-t_max, t_max-dt, t_num)
freq = np.linspace(-freq_max, freq_max-d_freq, t_num)


def abs_factor(n, m, gamma, f, w_0, w_eg, d):
    return (1.j/(np.sqrt(2.*np.pi)*(1.j*gamma + f + (m-n)*w_0 - w_eg)))*overlap(d, n, m, w_0)**2


def spectra(n, m, gamma, f, w_0, w_eg, d):
    sum_spectra = np.zeros(f.size, dtype=np.complex)
    for i in range(n):
        for j in range(m-i):
            sum_spectra += abs_factor(i+j, i, gamma, f, w_0, w_eg, d)
    return sum_spectra

n_max = 4
m_max = 4
#
w_eg_Pr = .454
# w_01 = np.linspace(0.03, 0.13, 11)
# gamma1 = 1./np.linspace(15., 75., 11)
# d1 = np.linspace(0.2, 1., 11)
#
data = np.loadtxt("Cph8_RefCrossSect.csv", delimiter=',')
data[:, 1] /= data[:, 1].max()
#
#
# def spectra_fit(params):
#
#     w_0 = params[0]
#     gamma = params[1]
#     d = params[2]
#
#     Pr_fit_data = \
#         (np.exp(-w_0 / kT)*freq[t_num/2+1:]*spectra(m_max, n_max, gamma, freq[t_num/2+1:], w_0, w_eg_Pr, d)).real
#     Pr_fit_data /= Pr_fit_data.max()
#     return norm(Pr_fit_data - data[:, 1], 1)
#
# from itertools import product
# from multiprocessing import Pool
# import time
# start = time.time()
# result = np.asarray(Pool(4).map(spectra_fit, product(w_01, gamma1, d1))).reshape(11, 11, 11)
# end = time.time()
#
# print end - start, 'Total time elapsed'
# print result.min()
# key = np.unravel_index(result.argmin(), result.shape)
# print key


def spectra_check(params):
    w0 = params[0]
    gamma = params[1]
    d1 = params[2]
    # print spectra_fit((w0, gamma, d1))

    Pr_best_fit = \
            (np.exp(-w0/ kT)*freq[t_num/2+1:]*spectra(m_max, n_max, gamma, freq[t_num/2+1:], w0, w_eg_Pr, d1).real)
    Pr_best_fit /= Pr_best_fit.max()

    plt.figure()
    plt.suptitle("Cph1 PR state spectra")
    plt.plot((660.*.454)/(freq[t_num/2+1:]), Pr_best_fit, 'k', label='Pr_calc')
    plt.plot(data[:, 0], data[:, 1], 'r-.', label='Pr_exp')
    # plt.plot(data[:, 0], data[:, 2], 'k-.', label='Pfr_exp')
    plt.xlim(450., 900.0)
    plt.xlabel("Wavelength (in nm)")
    plt.ylabel("Normalized absorbtion spectra")
    plt.grid()
    plt.legend()
    plt.show()

spectra_check((0.0185, 1./70, .35))

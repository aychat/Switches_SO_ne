import numpy as np
from numpy import fft
from scipy import fftpack, linalg
from types import MethodType, FunctionType  # this is used to dynamically add method to class
import numexpr as ne


class RhoPropagate:
    """
    Calculates rho(t) given rho(0) using the split-operator method for 
    the Hamiltonian H = p^2/2 + V(x) with V(x) = (1/2)*(omega*x)^2 
    =================================================================
    """

    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            Vg(x) - the ground electronic state adiabatic potential curve
            Ve(x) - the first excited electronic state adiabatic potential curve
            Vge(x, t) - coupling between ground and excited states via laser-molecule interaction
            K(p) - kinetic energy
            dt - time step
            t0 (optional) - initial value of time
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

            # Check that all attributes were specified

        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert 2**int(np.log2(self.X_gridDIM)) == self.X_gridDIM, "Coordinate grid not a power of 2"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.dt
        except AttributeError:
            raise AttributeError("Time-step (dt) was not specified")

        try:
            self.t
        except AttributeError:
            print("Warning: Initial time (t) was not specified, thus it is set to zero.")
            self.t = 0.

        try:
            self.abs_boundary
        except AttributeError:
            print("Warning: Absorbing boundary (abs_boundary) was not specified, thus it is turned off")
            self.abs_boundary = 1.

        # coordinate step size
        self.dX = 2. * self.X_amplitude / self.X_gridDIM
        self.dP = np.pi / self.X_amplitude
        self.P_amplitude = self.dP * self.X_gridDIM / 2.

        # grid for bra and kets
        self.Xrange = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)
        self.Prange = np.linspace(-self.P_amplitude, self.P_amplitude - self.dP, self.X_gridDIM)

        # generate coordinate grid
        self.X1 = self.Xrange[:, np.newaxis]
        self.X2 = self.Xrange[np.newaxis, :]

        # generate momentum grid
        self.P1 = fft.fftshift(self.Prange[:, np.newaxis])
        self.P2 = fft.fftshift(self.Prange[np.newaxis, :])

        # Pre-calculate the potential energy phase
        # self.expV = ne.evaluate(self.VgX.format(x='X2'), local_dict=self.__dict__) \
        #             - ne.evaluate(self.VgX.format(x='X1'), local_dict=self.__dict__) \
        #             + ne.evaluate(self.VeX.format(x='X2'), local_dict=self.__dict__) \
        #             - ne.evaluate(self.VeX.format(x='X1'), local_dict=self.__dict__)


        self.expV = ne.evaluate(
                'exp(0.5j * dt * ('
                + self.VgX.format(x='X2') + '-'
                + self.VgX.format(x='X1') + '+'
                + self.VeX.format(x='X2') + '-'
                + self.VeX.format(x='X1') +
                '))',
                local_dict=self.__dict__
        )

        # CHANEGE THIS
        # Pre-calculate the kinetic energy phase
        # self.expK = ne.evaluate(self.KP.format(p='P2'), local_dict=self.__dict__)\
        #             - ne.evaluate(self.KP.format(p='P1'), local_dict=self.__dict__)
        # self.expK = 1j * self.dt * self.expK
        # ne.evaluate("exp(expK)", local_dict=self.__dict__, out=self.expK)

        self.expK = ne.evaluate(
            'exp(1j * dt * ('
            + self.KP.format(p='P2') + '-'
            + self.KP.format(p='P1') +
            '))',
            local_dict=self.__dict__
        )

        # Generate codes
        self.Vg_minus_Ve_X1 = ne.evaluate(self.VgX.format(x='X1') + '-' + self.VeX.format(x='X1'),
                                          local_dict=self.__dict__)
        self.Vg_minus_Ve_X2 = ne.evaluate(self.VgX.format(x='X2') + '-' + self.VeX.format(x='X2'),
                                          local_dict=self.__dict__)


    def get_CML_matrices(self, q, t):
        # assert q is self.X1 or q is self.X2, "Either X1 or X2 expected as coordinate"

        Vge = self.Vge(q, t)

        self.Vg_minus_Ve = (self.Vg_minus_Ve_X1 if q is self.X1 else self.Vg_minus_Ve_X2)

        D = np.sqrt(Vge ** 2 + 0.25 * self.Vg_minus_Ve_X1 ** 2)
        S = np.sin(D * self.dt)
        S /= D

        C = np.cos(D * self.dt)

        M = 0.5 * S * self.Vg_minus_Ve

        L = S * Vge

        return C, M, L

    def get_T_left(self, t):

        C, M, L = self.get_CML_matrices(self.X1, t)
        return C-1j*M, -1j*L, C+1j*M

    def get_T_right(self, t):

        C, M, L = self.get_CML_matrices(self.X2, t)
        return C+1j*M, 1j*L, C-1j*M

    def get_T_left_A(self, t):

        C, M, L = self.get_CML_matrices(self.X1, t)
        return C + 1j * M, 1j * L, C - 1j * M

    def get_T_right_A(self, t):

        C, M, L = self.get_CML_matrices(self.X2, t)
        return C - 1j * M, -1j * L, C + 1j * M

    def Vge(self, q, t):
        return eval(self.codeVge)

    def field(self, t):
        return eval(self.codefield)


    def slice(self, *args):
        """
        Slice array A listed in args as if they were returned by fft.rfft(A, axis=0)
        """
        return (A[:(1 + self.X_gridDIM//2), :] for A in args)

    def single_step_propagation(self, time):
        """
        Perform single step propagation
        """

        # Construct T matrices
        TgL, TgeL, TeL = self.get_T_left(time)
        TgR, TgeR, TeR = self.get_T_right(time)

        # Save previous version of the density matrix
        rhoG, rhoGE, rhoGE_c, rhoE = self.rho_g, self.rho_ge, self.rho_ge_c, self.rho_e

        # First update the complex valued off diagonal density matrix
        self.rho_ge = (TgL*rhoG + TgeL*rhoGE_c)*TgeR + (TgL*rhoGE + TgeL*rhoE)*TeR
        self.rho_ge_c = (TgeL*rhoG + TeL*rhoGE_c)*TgR + (TgeL*rhoGE + TeL*rhoE)*TgeR

        # Slice arrays to employ the symmetry (savings in speed)
        # TgL, TgeL, TeL = self.slice(TgL, TgeL, TeL)
        # TgR, TgeR, TeR = self.slice(TgR, TgeR, TeR)
        # rhoG, rhoGE, rhoE = self.slice(rhoG, rhoGE, rhoE)

        # Calculate the remaining real valued density matrix
        self.rho_g = (TgL*rhoG + TgeL*rhoGE_c)*TgR + (TgL*rhoGE + TgeL*rhoE)*TgeR
        self.rho_e = (TgeL*rhoG + TeL*rhoGE_c)*TgeR + (TgeL*rhoGE + TeL*rhoE)*TeR

        # ---------- Apply kinetic phase factor ------------ #
        self.rho_ge *= self.expV
        self.rho_ge_c *= self.expV
        self.rho_g *= self.expV  # [:(1 + self.X_gridDIM//2), :]
        self.rho_e *= self.expV  # [:(1 + self.X_gridDIM//2), :]

        # --------------- x1 x2  ->  p1 x2 ----------------- #
        self.rho_ge = fftpack.ifft(self.rho_ge, axis=0, overwrite_x=True)
        self.rho_ge_c = fftpack.ifft(self.rho_ge_c, axis=0, overwrite_x=True)
        self.rho_g = fftpack.ifft(self.rho_g, axis=0, overwrite_x=True)
        self.rho_e = fftpack.ifft(self.rho_e, axis=0, overwrite_x=True)

        # --------------- p1 x2  ->  p1 p2 ----------------- #
        self.rho_ge = fftpack.fft(self.rho_ge, axis=1, overwrite_x=True)
        self.rho_ge_c = fftpack.fft(self.rho_ge_c, axis=1, overwrite_x=True)
        self.rho_g = fftpack.fft(self.rho_g, axis=1, overwrite_x=True)
        self.rho_e = fftpack.fft(self.rho_e, axis=1, overwrite_x=True)

        # ---------- Apply kinetic phase factor ------------ #
        self.rho_ge *= self.expK
        self.rho_ge_c *= self.expK
        self.rho_g *= self.expK  # [:, :(1 + self.X_gridDIM//2)]
        self.rho_e *= self.expK  # [:, :(1 + self.X_gridDIM//2)]

        # --------------- p1 p2  ->  p1 x2 ----------------- #
        self.rho_ge = fftpack.ifft(self.rho_ge, axis=1, overwrite_x=True)
        self.rho_ge_c = fftpack.ifft(self.rho_ge_c, axis=1, overwrite_x=True)
        self.rho_g = fftpack.ifft(self.rho_g, axis=1, overwrite_x=True)
        self.rho_e = fftpack.ifft(self.rho_e, axis=1, overwrite_x=True)

        # --------------- p1 x2  ->  x1 x2 ----------------- #
        self.rho_ge = fftpack.fft(self.rho_ge, axis=0, overwrite_x=True)
        self.rho_ge_c = fftpack.fft(self.rho_ge_c, axis=0, overwrite_x=True)
        self.rho_g = fftpack.fft(self.rho_g, axis=0, overwrite_x=True)
        self.rho_e = fftpack.fft(self.rho_e, axis=0, overwrite_x=True)
        self.normalize_rho()

    def single_step_propagation_A_inverse(self, rhoA_g, rhoA_ge, rhoA_ge_c, rhoA_e, time):
        """
        Perform single step propagation
        """

        # Construct T matrices
        TgL, TgeL, TeL = self.get_T_left_A(time)
        TgR, TgeR, TeR = self.get_T_right_A(time)

        # Save previous version of the density matrix
        rhoG_A, rhoGE_A, rhoGE_c_A, rhoE_A = rhoA_g, rhoA_ge, rhoA_ge_c, rhoA_e

        # First update the complex valued off diagonal density matrix
        rhoA_ge = (TgL*rhoG_A + TgeL*rhoGE_c_A)*TgeR + (TgL*rhoGE_A + TgeL*rhoE_A)*TeR
        rhoA_ge_c = (TgeL*rhoG_A + TeL*rhoGE_c_A)*TgR + (TgeL*rhoGE_A + TeL*rhoE_A)*TgeR

        # Slice arrays to employ the symmetry (savings in speed)
        # TgL, TgeL, TeL = self.slice(TgL, TgeL, TeL)
        # TgR, TgeR, TeR = self.slice(TgR, TgeR, TeR)
        # rhoG, rhoGE, rhoE = self.slice(rhoG, rhoGE, rhoE)

        # Calculate the remaining real valued density matrix
        rhoA_g = (TgL*rhoG_A + TgeL*rhoGE_c_A)*TgR + (TgL*rhoGE_A + TgeL*rhoE_A)*TgeR
        rhoA_e = (TgeL*rhoG_A + TeL*rhoGE_c_A)*TgeR + (TgeL*rhoGE_A + TeL*rhoE_A)*TeR

        # ---------- Apply kinetic phase factor ------------ #
        rhoA_ge *= self.expV.conj()
        rhoA_ge_c *= self.expV.conj()
        rhoA_g *= self.expV.conj()  # [:(1 + self.X_gridDIM//2), :]
        rhoA_e *= self.expV.conj() # [:(1 + self.X_gridDIM//2), :]

        # --------------- x1 x2  ->  p1 x2 ----------------- #
        rhoA_ge = fftpack.ifft(rhoA_ge, axis=0, overwrite_x=True)
        rhoA_ge_c = fftpack.ifft(rhoA_ge_c, axis=0, overwrite_x=True)
        rhoA_g = fftpack.ifft(rhoA_g, axis=0, overwrite_x=True)
        rhoA_e = fftpack.ifft(rhoA_e, axis=0, overwrite_x=True)

        # --------------- p1 x2  ->  p1 p2 ----------------- #
        rhoA_ge = fftpack.fft(rhoA_ge, axis=1, overwrite_x=True)
        rhoA_ge_c = fftpack.fft(rhoA_ge_c, axis=1, overwrite_x=True)
        rhoA_g = fftpack.fft(rhoA_g, axis=1, overwrite_x=True)
        rhoA_e = fftpack.fft(rhoA_e, axis=1, overwrite_x=True)

        # ---------- Apply kinetic phase factor ------------ #
        rhoA_ge *= self.expK.conj()
        rhoA_ge_c *= self.expK.conj()
        rhoA_g *= self.expK.conj()  # [:, :(1 + self.X_gridDIM//2)]
        rhoA_e *= self.expK.conj()  # [:, :(1 + self.X_gridDIM//2)]

        # --------------- p1 p2  ->  p1 x2 ----------------- #
        rhoA_ge = fftpack.ifft(rhoA_ge, axis=1, overwrite_x=True)
        rhoA_ge_c = fftpack.ifft(rhoA_ge_c, axis=1, overwrite_x=True)
        rhoA_g = fftpack.ifft(rhoA_g, axis=1, overwrite_x=True)
        rhoA_e = fftpack.ifft(rhoA_e, axis=1, overwrite_x=True)

        # --------------- p1 x2  ->  x1 x2 ----------------- #
        rhoA_ge = fftpack.fft(rhoA_ge, axis=0, overwrite_x=True)
        rhoA_ge_c = fftpack.fft(rhoA_ge_c, axis=0, overwrite_x=True)
        rhoA_g = fftpack.fft(rhoA_g, axis=0, overwrite_x=True)
        rhoA_e = fftpack.fft(rhoA_e, axis=0, overwrite_x=True)

        # rhoA_g, rhoA_ge, rhoA_ge_c, rhoA_e = self.normalize_rho_A(rhoA_g, rhoA_ge, rhoA_ge_c, rhoA_e)
        # print "norm rho_g ", np.trace(self.rho_g), "norm rho_e ", np.trace(self.rho_e)
        return rhoA_g, rhoA_ge, rhoA_ge_c, rhoA_e

    def single_step_propagation_mu_rho(self, rhomu_g, rhomu_ge, rhomu_ge_c, rhomu_e, time):
        """
        Perform single step propagation
        """

        # Construct T matrices
        TgL, TgeL, TeL = self.get_T_left(time)
        TgR, TgeR, TeR = self.get_T_right(time)

        # Save previous version of the density matrix
        rhomuG, rhomuGE, rhomuGE_c, rhomuE = rhomu_g, rhomu_ge, rhomu_ge_c, rhomu_e

        # First update the complex valued off diagonal density matrix
        rhomu_ge = (TgL*rhomuG + TgeL*rhomuGE_c)*TgeR + (TgL*rhomuGE + TgeL*rhomuE)*TeR
        rhomu_ge_c = (TgeL*rhomuG + TeL*rhomuGE_c)*TgR + (TgeL*rhomuGE + TeL*rhomuE)*TgeR

        # Slice arrays to employ the symmetry (savings in speed)
        # TgL, TgeL, TeL = self.slice(TgL, TgeL, TeL)
        # TgR, TgeR, TeR = self.slice(TgR, TgeR, TeR)
        # rhoG, rhoGE, rhoE = self.slice(rhoG, rhoGE, rhoE)

        # Calculate the remaining real valued density matrix
        rhomu_g = (TgL*rhomuG + TgeL*rhomuGE_c)*TgR + (TgL*rhomuGE + TgeL*rhomuE)*TgeR
        rhomu_e = (TgeL*rhomuG + TeL*rhomuGE_c)*TgeR + (TgeL*rhomuGE + TeL*rhomuE)*TeR

        # ---------- Apply kinetic phase factor ------------ #
        rhomu_ge *= self.expV
        rhomu_ge_c *= self.expV
        rhomu_g *= self.expV  # [:(1 + self.X_gridDIM//2), :]
        rhomu_e *= self.expV  # [:(1 + self.X_gridDIM//2), :]

        # --------------- x1 x2  ->  p1 x2 ----------------- #
        rhomu_ge = fftpack.ifft(rhomu_ge, axis=0, overwrite_x=True)
        rhomu_ge_c = fftpack.ifft(rhomu_ge_c, axis=0, overwrite_x=True)
        rhomu_g = fftpack.ifft(rhomu_g, axis=0, overwrite_x=True)
        rhomu_e = fftpack.ifft(rhomu_e, axis=0, overwrite_x=True)

        # --------------- p1 x2  ->  p1 p2 ----------------- #
        rhomu_ge = fftpack.fft(rhomu_ge, axis=1, overwrite_x=True)
        rhomu_ge_c = fftpack.fft(rhomu_ge_c, axis=1, overwrite_x=True)
        rhomu_g = fftpack.fft(rhomu_g, axis=1, overwrite_x=True)
        rhomu_e = fftpack.fft(rhomu_e, axis=1, overwrite_x=True)

        # ---------- Apply kinetic phase factor ------------ #
        rhomu_ge *= self.expK
        rhomu_ge_c *= self.expK
        rhomu_g *= self.expK  # [:, :(1 + self.X_gridDIM//2)]
        rhomu_e *= self.expK  # [:, :(1 + self.X_gridDIM//2)]

        # --------------- p1 p2  ->  p1 x2 ----------------- #
        rhomu_ge = fftpack.ifft(rhomu_ge, axis=1, overwrite_x=True)
        rhomu_ge_c = fftpack.ifft(rhomu_ge_c, axis=1, overwrite_x=True)
        rhomu_g = fftpack.ifft(rhomu_g, axis=1, overwrite_x=True)
        rhomu_e = fftpack.ifft(rhomu_e, axis=1, overwrite_x=True)

        # --------------- p1 x2  ->  x1 x2 ----------------- #
        rhomu_ge = fftpack.fft(rhomu_ge, axis=0, overwrite_x=True)
        rhomu_ge_c = fftpack.fft(rhomu_ge_c, axis=0, overwrite_x=True)
        rhomu_g = fftpack.fft(rhomu_g, axis=0, overwrite_x=True)
        rhomu_e = fftpack.fft(rhomu_e, axis=0, overwrite_x=True)

        # rhomu_g, rhomu_ge, rhomu_ge_c, rhomu_e = self.normalize_rho_A(rhomu_g, rhomu_ge, rhomu_ge_c, rhomu_e)
        # print "norm rho_g ", np.trace(self.rho_g), "norm rho_e ", np.trace(self.rho_e)
        return rhomu_g, rhomu_ge, rhomu_ge_c, rhomu_e

    def normalize_rho(self):
        # norm = (self.rho_g.sum() + self.rho_e.sum()) * self.dX
        norm = np.trace(self.rho_g) + np.trace(self.rho_e)
        self.rho_e /= norm
        self.rho_g /= norm
        self.rho_ge /= norm
        self.rho_ge_c /= norm

    def normalize_rho_A(self, rhoA_g, rhoA_ge, rhoA_ge_c, rhoA_e):
        # norm = (rhoA_g.sum() + rhoA_e.sum()) * self.dX
        norm = np.trace(rhoA_g) + np.trace(rhoA_e)
        rhoA_e /= norm
        rhoA_g /= norm
        rhoA_ge /= norm
        rhoA_ge_c /= norm

        return rhoA_g, rhoA_ge, rhoA_ge_c, rhoA_e

    def set_initial_rho(self, rhoG=0, rhoGE=0, rhoGE_c=0, rhoE=0):
        shape = (self.X_gridDIM, self.X_gridDIM)
        self.rho_g = (rhoG if isinstance(rhoG, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_e = (rhoE if isinstance(rhoE, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_ge = (rhoGE if isinstance(rhoGE, np.ndarray) else np.zeros(shape, dtype=np.complex))
        self.rho_ge_c = (rhoGE_c if isinstance(rhoGE_c, np.ndarray) else np.zeros(shape, dtype=np.complex))
        return self

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # Use the documentation string for the developed class
    print(RhoPropagate.__doc__)

    from Calculate_gibbs_state import SplitOpRho

    qsys_params = dict(
        t=0.,
        dt=0.01,

        X_gridDIM=128,
        X_amplitude=10.,

        kT=0.1,
        Tsteps=500,
        field_sigma2=2 * .6 ** 2,
        gamma=0.005,

        # kinetic energy part of the hamiltonian
        KP="0.5*{p}**2",

        freq_Vg=1.075,
        freq_Ve=1.075,
        disp=1.,
        Ediff=9.,
        delt=0.75,

        # potential energy part of the hamiltonian
        VgX="0.5*(freq_Vg*{x})**2",
        VeX="0.5*(freq_Ve*({x}-disp))**2 + Ediff",

        codeVge="-.05*q*self.field(t)",
        codedipole=".05*q",
        codefield="50.*np.exp(-(1./self.field_sigma2)*(t - 0.5*self.dt*self.Tsteps)**2)*np.cos(self.delt*self.Ediff*t)"
    )

    import time
    start = time.time()

    molecule = RhoPropagate(**qsys_params)
    gibbs_state = SplitOpRho(**qsys_params).get_gibbs_state()

    molecule.set_initial_rho(gibbs_state)

    t = np.linspace(0.0, molecule.Tsteps*molecule.dt, molecule.Tsteps)
    plt.figure()
    plt.plot(t, molecule.field(t))

    plt.figure()
    plt.suptitle("Time evolution of the density operator")
    plt.subplot(211)
    plt.title("$\\rho_0$")
    plt.plot(molecule.Xrange, np.diag(molecule.rho_g).real, 'r', label='Initial ground state')
    plt.plot(molecule.Xrange, np.diag(molecule.rho_e).real, 'k', label='Initial excited state')
    plt.xlabel("x")
    plt.ylabel("$\\rho(x, x, 0)$")
    plt.legend()
    plt.grid()

    # from plot_functions import animate_1d_subplots, animate_2d_imshow, plot_2d_subplots
    for _ in range(molecule.Tsteps):
        molecule.single_step_propagation(molecule.t)
        molecule.t += molecule.dt

    plt.subplot(212)
    plt.title("$\\rho_T$")
    plt.plot(molecule.Xrange, np.diag(molecule.rho_g).real, 'r', label='Final ground state')
    plt.plot(molecule.Xrange, np.diag(molecule.rho_e).real, 'k', label='Final excited state')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("$\\rho(x, x, T)$")
    plt.grid()

    print np.trace(molecule.rho_g)
    print np.trace(molecule.rho_e)

    print molecule.rho_ge.sum()*molecule.dX
    print molecule.rho_ge_c.sum()*molecule.dX

    end = time.time()
    print
    print end - start
    plt.show()







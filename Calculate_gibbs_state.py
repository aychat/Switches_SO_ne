import numpy as np
import numexpr as ne
# numpy.fft has better implementation of real fourier transform
# necessary for real split operator propagator
from numpy import fft
from scipy import fftpack


class SplitOpRho:
    """
    The second-order split-operator propagator for
    finding the Wigner function of the Maxwell-Gibbs canonical state [rho = exp(-H/kT)]
    by split-operator propagation of the Bloch equation in phase space.
    The Hamiltonian should be of the form H = K(p) + V(x).

    This implementation follows split_op_wigner_moyal.py
    """

    def __init__(self, **kwargs):
        """
        The following parameters must be specified
            X_gridDIM - the coordinate grid size
            X_amplitude - maximum value of the coordinates
            P_gridDIM - the momentum grid size
            P_amplitude - maximum value of the momentum
            V(x) - potential energy (as a function)
            K(p) - momentum dependent part of the hamiltonian (as a function)
            kT  - the temperature (if kT = 0, then the ground state Wigner function will be obtained.)
            dbeta - (optional) 1/kT step size
        """

        # save all attributes
        for name, value in kwargs.items():
            setattr(self, name, value)

        # Check that all attributes were specified
        try:
            self.X_gridDIM
        except AttributeError:
            raise AttributeError("Coordinate grid size (X_gridDIM) was not specified")

        assert self.X_gridDIM % 2 == 0, "Coordinate grid size (X_gridDIM) must be even"

        try:
            self.X_amplitude
        except AttributeError:
            raise AttributeError("Coordinate grid range (X_amplitude) was not specified")

        try:
            self.Vg
        except AttributeError:
            raise AttributeError("Potential energy (V) was not specified")

        try:
            self.K
        except AttributeError:
            raise AttributeError("Momentum dependence (K) was not specified")

        try:
            self.kT
        except AttributeError:
            raise AttributeError("Temperature (kT) was not specified")

        if self.kT > 0:
            try:
                self.dbeta
            except AttributeError:
                # if dbeta is not defined, just choose some value
                self.dbeta = 0.01

            # get number of dbeta steps to reach the desired Gibbs state
            self.num_beta_steps = 1. / (self.kT * self.dbeta)

            if round(self.num_beta_steps) != self.num_beta_steps:
                # Changing self.dbeta so that num_beta_steps is an exact integer
                self.num_beta_steps = round(self.num_beta_steps)
                self.dbeta = 1. / (self.kT * self.num_beta_steps)

            self.num_beta_steps = int(self.num_beta_steps)
        else:
            raise NotImplemented("The calculation of the ground state Wigner function has not been implemented")

        ###################################################################################
        #
        #   Generate grids
        #
        ###################################################################################

        # get coordinate and momentum step sizes
        self.dX = 2. * self.X_amplitude / self.X_gridDIM

        # coordinate grid
        self.X = np.linspace(-self.X_amplitude, self.X_amplitude - self.dX, self.X_gridDIM)
        self.X1 = self.X[np.newaxis, :]
        self.X2 = self.X[:, np.newaxis]

        # Lambda grid (variable conjugate to the coordinate)
        self.P = fft.fftfreq(self.X_gridDIM, self.dX / (2 * np.pi))
        self.P1 = self.P[np.newaxis, :]
        self.P2 = self.P[:, np.newaxis]

        ###################################################################################
        #
        # Pre-calculate exponents used for the split operator propagation
        #
        ###################################################################################

        # Get the sum of the potential energy contributions
        self.expV = ne.evaluate(self.VgX1, local_dict=self.__dict__) + ne.evaluate(self.VgX2, local_dict=self.__dict__)
        self.expV *= -0.25 * self.dbeta

        # Make sure that the largest value is zero
        self.expV -= self.expV.max()
        # such that the following exponent is never larger then one
        np.exp(self.expV, out=self.expV)

        # Get the sum of the kinetic energy contributions
        self.expK = ne.evaluate(self.KP1, local_dict=self.__dict__) + ne.evaluate(self.KP2, local_dict=self.__dict__)
        self.expK *= -0.5 * self.dbeta

        # Make sure that the largest value is zero
        self.expK -= self.expK.max()
        # such that the following exponent is never larger then one
        np.exp(self.expK, out=self.expK)

    def single_step_propagation(self):
        """
        Perform single step propagation. The final Wigner function is not normalized.
        :return: self.wignerfunction
        """
        self.rho *= self.expV

        # x1 x2  ->  p1 x2
        self.rho = fftpack.ifft(self.rho, axis=0)

        # p1 x2  ->  p1 p2
        self.rho = fftpack.fft(self.rho, axis=1)
        self.rho *= self.expK

        # p1 p2  ->  p1 x2
        self.rho = fftpack.ifft(self.rho, axis=1)

        # p1 x2  -> x1 x2
        self.rho = fftpack.fft(self.rho, axis=0)
        self.rho *= self.expV

        return self.rho

    def Vg(self, q):
        return eval(self.codeVg)

    def K(self, p):
        return eval(self.codeK)

    def get_gibbs_state(self):
        """
        Calculate the Boltzmann-Gibbs state and save it in self.gibbs_state
        :return:
        """

        # Initialize the Wigner function as the infinite temperature Gibbs state
        self.rho = 1.

        for _ in xrange(self.num_beta_steps):
            # propagate by dbeta
            self.single_step_propagation()

            # normalization
            self.rho /= np.trace(self.rho)

        return self.rho


##############################################################################
#
#   Run some examples
#
##############################################################################

if __name__ == '__main__':
    print(SplitOpRho.__doc__)

    import matplotlib.pyplot as plt

    qsys_params = dict(
        t=0.,
        dt=0.01,

        X_gridDIM=128,
        X_amplitude=10.,

        kT=0.1,
        Tsteps=500,
        field_sigma2=2 * .6 ** 2,
        gamma=0.5,

        # kinetic energy part of the hamiltonian
        KP1="0.5*P1**2",
        KP2="0.5*P2**2",
        freq_Vg=1.075,
        freq_Ve=1.075,
        disp=1.,
        Ediff=9.,
        delt=0.75,

        # potential energy part of the hamiltonian
        VgX1="0.5*(freq_Vg*X1)**2",
        VgX2="0.5*(freq_Vg*X2)**2",

        VeX1="0.5*(freq_Ve*(X1-disp))**2 + Ediff",
        VeX2="0.5*(freq_Ve*(X2-disp))**2 + Ediff"
    )

    print("Calculating the Gibbs state...")
    molecule = SplitOpRho(**qsys_params)
    gibbs_state = molecule.get_gibbs_state().real

    ##############################################################################
    #
    #   Plot the results
    #
    ##############################################################################

    from Normalization import RhoSymLogNorm, RhoNormalize

    # save common plotting parameters
    plot_params = dict(
        origin='lower',
        extent=[molecule.X1.min(), molecule.X1.max(), molecule.X2.min(), molecule.X2.max()],
        cmap='seismic',
        # norm=RhoSymLogNorm(linthresh=1e-13, vmin=-0.01, vmax=0.1)
        norm=RhoNormalize(vmin=-0.01, vmax=0.1)
    )

    plt.title("The Gibbs state (initial state)")
    plt.imshow(gibbs_state, **plot_params)
    plt.colorbar()
    plt.xlabel('$x$ (a.u.)')
    plt.ylabel('$x^{\'}$ (a.u.)')
    plt.show()

    plt.plot(molecule.X, np.diag(gibbs_state))
    plt.show()
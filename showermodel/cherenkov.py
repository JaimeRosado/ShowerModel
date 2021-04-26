# coding: utf-8

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)


# Constructor
def Cherenkov(profile):
    """
    Calculate the Cherenkov light production from a shower profile.
    
    The parameterization described in F. Nerling et al., Astropart. Phys.
    24(2006)241 is used.

    Parameters
    ----------
    profile : Profile object.

    Returns
    -------
    cherenkov : Cherenkov object.
    """
    cherenkov = _Cherenkov(columns=['N_ph', 'a', 'theta_c', 'b', 'theta_cc'])
    cherenkov.profile = profile
    cherenkov.atmosphere = profile.atmosphere
    C_E = _E_factor(profile.s, profile.atmosphere.E_th, profile.E/2.)
    cherenkov.N_ph = (2. * np.pi / 137.036 * (1. / 290. - 1. / 430.)
                      * 1e12 * C_E * profile.N_ch * profile.dl)

    cherenkov.a = 0.42489 + 0.58371 * profile.s - 0.082373 * profile.s**2
    cherenkov.theta_c = np.degrees(0.62694 / profile.atmosphere.E_th**0.6059)
    cherenkov.b = 0.055108 - 0.095587 * profile.s + 0.056952 * profile.s**2
    theta_cc = np.array((10.509 - 4.9444 * profile.s) * cherenkov.theta_c)
    theta_cc[theta_cc < 0.] = 0.
    cherenkov.theta_cc = theta_cc

    return cherenkov


# Class #######################################################################
class _Cherenkov(pd.DataFrame):
    """
    DataFrame containing the Cherenkov ligth production.
    
    The Cherenkov ligh is evaluated in the 290 - 430 nm range. The DataFrame
    includes the parameters determining the angular distribution of
    Cherenkov emission based on the parameterization described in
    F. Nerling et al., Astropart. Phys. 24(2006)241.

    Columns
    -------
    N_ph : int
        Number of Cherenkov photons in the 290 - 430 nm range.
    a : float
        First parameter of the angular distribution of Cherenkov emission.
    theta_c : float
        Second parameter (degrees) of the angular distribution.
    b : float
        Third parameter of the angular distribution.
    theta_cc : float
        Fourth parameter (degrees) of the angular distribution.

    Atributes
    ---------
    profile : Profile object.
    atmosphere : Atmosphere object.

    Methods
    -------
    show : Show the production of Cherenkov photons in the 290 - 430 nm range.
    as a function of slant depth.
    """
    def _E_factor(self, s, Eth, Emax):
        """
        Calculate a factor enclosing the dependence of the Cherenkov production
        on both the Cherenkov energy threshold E_th and the normalized energy
        distribution of electrons.
        """
        _E_factor(s, Eth, Emax)

    def show(self):
        """
        Show the production of Cherenkov photons in the 290 - 430 nm range as a
        function of slant depth.

        Returns
        -------
        ax : AxesSubplot object.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.plot(self.profile.X, self.N_ph, 'r-')
        ax.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax.axes.yaxis.set_label_text(
            "Cherenkov production (photons·cm$^2$/g)")
        return ax


# Auxiliary functions #########################################################
def _E_factor(s, Eth, Emax):
    """
    Calculate a factor enclosing the dependence of the Cherenkov production on
    both the Cherenkov energy threshold E_th and the normalized energy
    distribution of electrons, as they vary with the shower age:

    E_factor = (mc^2)^2 * int( ( 1/Eth^2 - 1/E^2 ) * f(E) * d(ln(E)) )
    where  f(E,s) = 1/N(s) * dN(E,s)/d(ln(E)),
    with int( f(E,s)*d(ln(E)) ) = 1
    See F. Nerling et al. Astropart. Phys. 24(2006)241.
    """
    # Parameterization of the energy distribution of secondary electrons as a
    # function of the shower age f(E,s)= a0(s) * E / [E+a1(s)] / [E+a2(s)]^s
    s = np.array(s)
    Eth = np.array(Eth)
    a0 = 0.145098 * np.exp(6.20114 * s - 0.596851 * s**2)
    a1 = 6.42522 - 1.53183 * s
    a2 = 168.168 - 42.1368 * s

    def tail(Eth, E, s):
        return E**(-s) * (1. / s / Eth**2 - 1. / (s + 2.) / E**2)

    Etail = min(np.exp(8.), Emax)  # exp(8.)=2981 MeV

    factor = np.zeros_like(Eth)
    for i in range(0, len(factor)):
        if Eth[i] > Emax:
            factor[i] = 0.
        elif Eth[i] >= Etail:  # Etail < Eth < Emax
            # For E > Etail: f(E,s) ~= a0/E^s
            # factor[i] = a0[i]*2./Eth[i]**(s[i]+2.)/s[i]/(s[i]+2.)
            factor[i] = a0[i] * (
                tail(Eth[i], Eth[i], s[i]) - tail(Eth[i], Emax, s[i]))
        else:        # Eth < Etail <= Emax
            lnE = np.arange(np.log(Eth[i]), 8., 0.1) + 0.05
            E = np.exp(lnE)
            # Normalized electron energy distribution:
            # f(E,s) = 1/N(s) * dN(E,s)/d(ln(E)),
            # with int( f(E,s)*d(ln(E)) ) = 1
            f = a0[i] * E / (E + a1[i]) / (E + a2[i])**s[i]
            factor[i] = np.sum((1. / Eth[i]**2 - 1. / E**2) * f) * 0.1
            # factor[i] = (factor[i] + a0[i] / Etail**s[i]
            # * (1. / s[i] / Eth[i]**2 - 1. / (s[i]+2.) / Etail**2))
            if Etail < Emax:
                factor[i] = factor[i] + a0[i] * (
                    tail(Eth[i], Etail, s[i]) - tail(Eth[i], Emax, s[i]))

    return 0.511**2 * factor

# coding: utf-8

import math
import numpy as np
import pandas as pd
import warnings
import showermodel as sm
import matplotlib.pyplot as plt
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)

# Default values for profile
_E = 10000000.  # MeV
from .track import _theta  # _theta = 0. degrees
_prf_model = 'Greisen'


# Constructor #################################################################
def Profile(E=_E, theta=_theta, alt=None, prf_model=_prf_model, X_max=None,
            X0_GH=None, lambda_GH=None, atmosphere=None, **kwargs):
    """
    Make a shower profile discretization.

    Parameters
    ----------
    E : float
        Energy of the primary particle in MeV.
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source. If None,
        theta is used. If given, theta is overwritten.
    prf_model : {'Greisen', 'Gaisser-Hillas'} or DataFrame
        If 'Greisen', the Greisen function for electromagnetic showers is used.
        If 'Gaisser-Hillas', the Gaisser-Hillas function for hadron-induced
        showers is used. If a DataFrame with an energy deposit profile is input,
        it must have two columns with the slant depth in g/cm2 and dE/dX in
        MeV.cm2/g. 
    X_max : float
        Slant depth in g/cm^2 at shower maximum. If None and prf_model is
        'Greisen' or 'Gaisser-Hillas', a typical value of X_max for gamma or
        proton showers is used. If None and a numerical energy deposit profile
        is input, lambda_r = 36.7 g/cm^2 is the radiation length and
        E_c = 81 MeV is the critical energy.
    X0_GH : float
        X0 parameter in g/cm2 to be used when prf_model=='Gaisser-Hillas'.
        If None, a typical value for the input energy is used.
    lambda_GH : float
        Lambda parameter in g/cm2 to be used when prf_model=='Gaisser-Hillas'.
        If None, a typical value for the input energy is used.
    atmosphere : Atmosphere object.
        If None, a new Atmosphere object is generated.
    **kwargs {h0, h_top, N_steps, model}
        Options to construct the new Atmosphere object when atmosphere==None.
        If None, the default Atmosphere object is used.

    Returns
    -------
    profile : Profile object.

    See also
    --------
    _Profile : Profile class.
    Shower : Contructor of Shower object.
    """
    from .atmosphere import _Atmosphere
    if isinstance(atmosphere, _Atmosphere):
        pass
    elif atmosphere is None:
        atmosphere = sm.Atmosphere(**kwargs)
    else:
        raise ValueError('The input atmosphere is not valid.')

    # The columns of the output DataFrame are: the slant depth in g/cm^2,
    # the shower age s, the energy deposit E_dep in MeV, and the number N_ch of
    # charged particles with energy above 1MeV
    profile = _Profile(columns=['X', 's', 'dX', 'E_dep', 'N_ch'])
    profile.atmosphere = atmosphere

    # The input shower parameters along with some geometric parameters are
    # included as atributes of the DataFrame
    profile.E = E
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    profile.theta = theta
    profile.alt = alt
    uz = np.cos(math.radians(theta))

    X = np.array(atmosphere.X_vert / uz)   # Slant depth in g/cm^2
    profile.X = X

    # Length in km travelled trhough one atmospheric slice
    profile.dl = atmosphere.h_step / uz
    # Depth in g/cm^2 travelled trhough one atmospheric slice
    profile.dX = 100000. * profile.dl * profile.atmosphere.rho

    profile.prf_model = prf_model

    N_ch = np.zeros_like(X)
    # DataFrame containing dE/dX at steps in X
    if isinstance(prf_model, pd.DataFrame):
        # Sorted to allow for interpolation
        prf_model.sort_index(axis=0, ascending=True, inplace=True)
        # The first column must be X in g/cm2
        X_model = np.array(prf_model.iloc[:, 0])
        # The second column must be dE_dX in MeV.cm2/g
        dE_dX_model = np.array(prf_model.iloc[:, 1])
        # Extreme values are added to allow for extrapolation
        X_model = np.insert(X_model, 0, X_model[0] * 1.5 - X_model[1] / 2.)
        X_model = np.append(X_model, X_model[-1] * 1.5 - X_model[-2] / 2.)
        dE_dX_model = np.insert(dE_dX_model, 0, 0.)
        dE_dX_model = np.append(dE_dX_model, 0.)

        index_max = dE_dX_model.argmax()  # Index where max of dE_dX is
        if X_max is None:
            X_max = X_model[index_max]
        elif (X_max < X_model[index_max-1]) or (X_max > X_model[index_max+1]):
            print("""
            Warning: The input X_max does not match the maximum of the input
            energy deposit profile.
            """)

        profile.X_max = X_max
        s = 3. * X / (X + 2. * X_max)  # Shower age
        profile.s = s

        profile.X0_GH = None
        profile.lambda_GH = None

        dE_dX = np.interp(X, X_model, dE_dX_model, left=0., right=0.)
        profile.E_dep = dE_dX * profile.dX
        profile.N_ch = dE_dX / profile._alpha(profile.s)

        if E < profile.E_dep.sum():
            profile.E = profile.E_dep.sum()
            raise ValueError("""
                The input shower energy is lower than the integraged energy
                deposit profile.
                """)
        elif E > profile.E_dep.sum()*1.2:
            print("""
            Warning: The input energy is greater than the integrated energy
            deposit profile by more than 20%.
            """)

    elif prf_model == 'Greisen':
        if X_max is None:
            # E_c=81 MeV is the critical energy in air
            X_max = 36.7 * np.log(E / 81.)
        profile.X_max = X_max
        s = 3. * X / (X + 2. * X_max)  # Shower age
        profile.s = s

        profile.X0_GH = None
        profile.lambda_GH = None

        # Greisen profile: N_ch = 0.31/sqrt(t_max) * exp[ t * (1-3/2*log(s)) ]
        # where t=X/lambda_r ,
        # lambda_r = 36.7 g/cm^2 is the radiation length in air
        N_ch[s > 0] = (0.31 / np.sqrt(X_max/36.7)
                       * np.exp(X[s > 0] / 36.7 * (1.-1.5*np.log(s[s > 0]))))
        # s>0 prevents from errors for discretization steps with X=0,
        # where the atmosphere is undefined

        # Shower size with an energy cut of 1MeV
        profile.N_ch = profile._Greisen_norm() * N_ch

        # Deposited energy in each slice: E_dep = alpha * N_ch * dX,
        # where  alpha(s) is the mean ionization energy loss per electron for
        # an energy cut of 1MeV
        profile.E_dep = profile._alpha(profile.s) * profile.N_ch * profile.dX

    elif prf_model == 'Gaisser-Hillas':
        # If not given, a typical value according to Heitler model is used
        if X_max is None:
            X_max = 36.7 * np.log(E / 4. / 81.)
            # lambda_r = 36.7 g/cm2 radiation lenght in air
            # E_c=81 MeV is the critical energy in air
        profile.X_max = X_max
        s = 3. * X / (X + 2. * X_max)  # Shower age
        profile.s = s

        # If not given, typical values according to Auger data are used
        if (X0_GH is None) or (lambda_GH is None):
            x = 6. + np.log10(E)
            R = 0.26 - 0.04 * (x - 17.9)
            L = 226.2 + 7.1 * (x - 17.9)
            if X0_GH is None:
                X0_GH = X_max - L / R
            if lambda_GH is None:
                lambda_GH = L * R
        profile.X0_GH = X0_GH
        profile.lambda_GH = lambda_GH

        # Gaisser-Hillas profile:
        # N_ch = N_ch_max * [ (X-X0) / (X_max-X0) ]**( (X_max-X0) / lambda )
        # * exp[ -(X-X_max) / lambda ]
        X_min = max(X0_GH, 0.)  # X0_GH is expected to be negative
        N_ch[X > X_min] = (
            ((X[X > X_min]-X0_GH) / (X_max-X0_GH))**((X_max-X0_GH)/lambda_GH)
            * np.exp(-(X[X > X_min]-X_max) / lambda_GH))
        # X>X_min prevents from errors for discretization steps with X=0,
        # where the atmosphere is undefined
        # or with X<X0_GH, where the GH profile is undefined

        # Shower size with an energy cut of 1MeV
        profile.N_ch = profile._GH_norm() * N_ch

        # Deposited energy in each slice: E_dep = alpha * N_ch * dX,
        # where  alpha(s) is the mean ionization energy loss per electron for
        # an energy cut of 1MeV
        profile.E_dep = profile._alpha(profile.s) * profile.N_ch * profile.dX

    else:
        raise ValueError('The input model is not valid.')

    return profile


# Class #######################################################################
class _Profile(pd.DataFrame):
    """
    DataFrame containing a shower profile discretization.

    Use sm.Profile to construct a Profile object.

    Columns
    -------
    X : float
        Slant depth in g/cm^2.
    s : float
        Shower age.
    dX : float
        Discretization step in g/cm^2 along the shower axis.
    E_dep : float
        Energy deposit in MeV at each discretiztion step.
    N_ch : float
        Number of charged particles.

    Attributes
    ----------
    atmosphere : Atmosphere object.
    E : float
        Energy of the primary particle.
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source.
    prf_model : {'Greisen', 'Gaisser-Hillas'} or DataFrame.
    X_max : float
        Slant depth in g/cm^2 at shower maximum.
    X0_GH : float
        X0 parameter in g/cm2 for prf_model=='Gaisser-Hillas'.
    lambda_GH : float
        lambda parameter in g/cm2 for prf_model=='Gaisser-Hillas'.
    dl : float
        Size in km of the discretization step along the shower axis.

    Methods
    -------
    Fluorescence : Calculate the fluorescence light production.
    Cherenkov : Calculate the Cherenkov light production.
    show : Show the shower profile, both number of charged particles and energy
        deposit, as a function of slant depth.

    See also
    --------
    Profile : Constructor of Profile object.
    Shower : Constructor of Shower object.
    """

    def _Greisen_norm(self):
        """
        Calculate the normalization constant K that relates a Greisen profile
        to the actual shower size N(s).
        """
        return _Greisen_norm(self.E, self.X_max)

    def _GH_norm(self):
        """
        Calculate the normalization constant K that relates a Gaisser-Hillas
        profile to the actual shower size N(s).
        """
        return _GH_norm(self.E, self.X_max, self.X0_GH, self.lambda_GH)

    def _alpha(self, s):
        """
        Calculate the mean ionization loss rate per electron in MeV/g.cm^2 as a
        function of shower age.
        """
        return _alpha(s)

    def Fluorescence(self):
        """
        Calculate the fluorescence photon production from a shower profile
        discretization.

        Returns
        -------
        Fluorescece object.
        """
        return sm.Fluorescence(self)

    def Cherenkov(self):
        """
        Calculate the Cherenkov light production from a shower profile
        discretization.

        Returns
        -------
        Cherenkov object.
        """
        return sm.Cherenkov(self)

    def show(self):
        """
        Show the shower profile, both number of charged particles and energy
        deposit, as a function of slant depth.

        Returns
        -------
        (ax1, ax2) : AxesSubplot objects.
        """
        # Shower size
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax1.plot(self.X, self.N_ch, 'r-')
        ax1.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax1.axes.yaxis.set_label_text("Number of charged particles")

        # Energy deposit
        ax2.plot(self.X, self.E_dep/self.dX, 'b-')
        ax2.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax2.axes.yaxis.set_label_text("Energy deposit (MeV·cm$^2$/g)")
        plt.tight_layout()
        return (ax1, ax2)


# Auxiliary functions #########################################################
def _Greisen_norm(E, X_max):
    """
    Calculate the normalization constant K that relates a Greisen profile to
    the actual shower size N(s) for an energy cut of 1MeV, that is,
    N(s) = K * N_g(s), with the following constraint:
    E = int[ dE/dX * dX ] = int[ alpha(s) * N(s) * dX/ds *ds ].
    """
    s = np.arange(0., 3., 0.01) + 0.005   # The shower age s ranges from 0 to 3
    X = 2. * X_max * s / (3. - s)         # slant depth
    # Greisen profile
    # N_g = 0.31/sqrt(X_max/lambda_r) * exp[ X/lambda_r * (1-3/2*log(s)) ] ,
    # lambda_r=36.7g/cm^2
    N_g = 0.31 / np.sqrt(X_max/36.7) * np.exp(X/36.7*(1.-1.5*np.log(s)))

    # Normalization constant K= E / int[alpha(s) * N_g(s) * dX/ds *ds]
    # where dX/ds = (X+2*X_max) / (3-s)
    return E / np.sum(_alpha(s) * N_g * (X+2.*X_max) / (3.-s) * 0.01)


# Must be developed
def _GH_norm(E, X_max, X0_GH, lambda_GH):
    """
    Calculate the normalization constant K that relates a Gaisser-Hillas
    profile to the actual shower size N(s) for an energy cut of 1MeV, that is,
    N(s) = K * N_GH(s), with the following constraint:
    E = int[ dE/dX * dX ] = int[ alpha(s) * N(s) * dX/ds *ds ].
    """
    s = np.arange(0., 3., 0.01) + 0.005   # The shower age s ranges from 0 to 3
    X = 2. * X_max * s / (3. - s)         # slant depth
    # Greisen profile
    # N_g = 0.31/sqrt(X_max/lambda_r) * exp[ X/lambda_r * (1-3/2*log(s)) ] ,
    # lambda_r=36.7g/cm^2
    # N_g = 0.31 / np.sqrt(X_max/36.7) * np.exp(X / 36.7 * (1.-1.5*np.log(s)))

    # Gaisser-Hillas profile:
    # N_ch = N_ch_max*[(X-X0)/(X_max-X0)]**((X_max-X0)/lambda)
    # *exp[-(X-X_max)/lambda]
    N_GH = (((X-X0_GH) / (X_max-X0_GH))**((X_max-X0_GH) / lambda_GH)
            * np.exp(-(X-X_max)/lambda_GH))

    # Normalization constant K= E / int[alpha(s) * N_g(s) * dX/ds *ds]
    # where dX/ds = (X+2*X_max) / (3-s)
    return E / np.sum(_alpha(s) * N_GH * (X+2.*X_max) / (3.-s) * 0.01)


def _alpha(s):
    """
    Calculate the mean ionization loss rate per electron in MeV/g.cm^2 as a
    function of shower age such that: dE/dX = alpha(s) * N(s)
    An energy cut of 1MeV on the shower electrons is assumed.
    F. Nerling et al., Astropart. Phys. 24(2006)241.
    """
    return 3.90883 / (1.05301+s)**9.91717 + 2.41715 + 0.13180 * s

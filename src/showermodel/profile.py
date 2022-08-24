# coding: utf-8

import numpy as np
import pandas as pd
import warnings
import showermodel as sm
import showermodel.constants as ct
import matplotlib.pyplot as plt
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)

# Default values for Profile
_Profile__E = ct.config['Shower']['E']
_Profile__theta = ct.config['Shower']['theta']
_Profile__alt = ct.config['Shower'].get('alt') # optional parameter
_Profile__zi = ct.config['Shower'].get('zi') # optional parameter
_Profile__prf_model = ct.config['Shower']['prf_model']
_Profile__X_max = ct.config['Shower'].get('X_max')  # optional parameter
_Profile__X0_GH = ct.config['Shower'].get('X0_GH')  # optional parameter
_Profile__lambda_GH = ct.config['Shower'].get('lambda_GH')  # optional parameter
_Profile__h0 = ct.config['Atmosphere']['h0']
_Profile__h_top = ct.config['Atmosphere'].get('h_top') # optional parameter
_Profile__N_steps = ct.config['Atmosphere']['N_steps']
_Profile__atm_model = ct.config['Atmosphere']['atm_model']
_Profile__rho_w_sl = ct.config['Atmosphere']['rho_w_sl']
_Profile__h_scale = ct.config['Atmosphere']['h_scale']


# Class #######################################################################
class Profile(pd.DataFrame):
    """
    DataFrame containing a shower profile discretization.

    Use sm.Profile() to construct the default Profile object.

    Parameters
    ----------
    E : float, default 10000000
        Energy of the primary particle in MeV.
    theta : float, default 0
        Zenith angle in degrees of the apparent position of the source.
    alt : float, default None
        Altitude in degrees of the apparent position of the source. If
        None, theta is used. If given, theta is overwritten.
    prf_model : 'Greisen', 'Gaisser-Hillas' or DataFrame, default 'Greisen'
        Profile model to be used. If 'Greisen', the Greisen function
        for electromagnetic showers is used. If 'Gaisser-Hillas', the
        Gaisser-Hillas function for hadron-induced showers is used.
        If a DataFrame is given, it should have two columns, the first
        one with the slant depth in g/cm2 and the second one with dE/dX
        in MeV.cm2/g.
    X_max : float, default None
        Slant depth in g/cm^2 at shower maximum. If None and prf_model
        is 'Greisen' or 'Gaisser-Hillas', a typical value of X_max for
        gamma or proton showers is calculated from the radiation length
        lambda_r = 36.7 g/cm^2 and the critical energy E_c = 81 MeV.
    X0_GH : float, default None
        X0 parameter in g/cm2 to be used when prf_model=='Gaisser-
        Hillas'. If None, a typical value for the input energy is used.
    lambda_GH : float, default None
        Lambda parameter in g/cm2 to be used when prf_model=='Gaisser-
        Hillas'. If None, a typical value for the input energy is used.
    zi : float, default None
        Height in km of the first interaction point of the shower. If
        None, the shower is assumed to begin at the top of the
        atmosphere (theta<90) or at ground level (theta>90).
    atmosphere : Atmosphere, default None
        Atmosphere object to be used. If None, a new Atmosphere object
        is generated. If given, h0, h_top, N_steps, atm_model, rho_w_sl
        and h_scale are ignored.
    h0 : float, default 0
        Ground level in km above sea level for the atmosphere
        discretization to be generated when atmosphere==None.
    h_top : float or None, default None
        Upper limit in km above sea level for the atmosphere
        discretization to be generated when atmosphere==None. If h_top
        is None, the top level of the selected atmospheric model is
        taken.
    N_steps : int, default 550
        Number of discretization steps for the atmosphere discretization
        to be generated when atmosphere==None.
    atm_model : int or DataFrame, default 1
        Atmospheric model used when atmosphere==None. If an int value
        is given, atm_model is searched from either CORSIKA atmospheric
        models (from 1 to 29) or a file named atm_models.toml in the
        working directory containing user-defined models. If a
        DataFrame is given, it should have two columns, one labelled as
        h with height in km and other labelled as X_vert or P,
        depending on whether vertical depth in g/cm^2 or pressure in
        hPa is given.
    rho_w_sl : float, default 7.5e-6
        Water-vapor density in g/cm^3 at sea level to calculate a
        simple exponential profile of water-vapor when
        atmosphere==None. Set to zero if dry air is assumed.
    h_scale : float, default 2.0
        Scale height in km to be used in the water-vapor exponential
        profile when atmospere==None.

    Attributes
    ----------
    X : float
        Column 0, slant depth in g/cm^2.
    s : float
        Column 1, shower age.
    dX : float
        Column 2, discretization step in g/cm^2 along the shower axis.
    E_dep : float
        Column 3, energy deposit in MeV at each discretization step.
    N_ch : float
        Column 4, number of charged particles.
    atmosphere : Atmosphere
        Atmosphere object.
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
    Fluorescence()
        Calculate the fluorescence light production.
    Cherenkov()
        Calculate the Cherenkov light production.
    show()
        Show the shower profile as a function of slant depth.

    See also
    --------
    Profile : DataFrame containing a shower profile discretization.
    Shower : Make a discretization of a shower.
    """
    def __init__(self, E=__E, theta=__theta, alt=__alt, prf_model=__prf_model,
                 X_max=__X_max, X0_GH=__X0_GH, lambda_GH=__lambda_GH, zi=__zi,
                 atmosphere=None, h0=__h0, h_top=__h_top, N_steps=__N_steps,
                 atm_model=__atm_model, rho_w_sl=__rho_w_sl,
                 h_scale=__h_scale):
        super().__init__(columns=['X', 's', 'dX', 'E_dep', 'N_ch'])
        _profile(self, E, theta, alt, prf_model, X_max, X0_GH, lambda_GH, zi,
                 atmosphere, h0, h_top, N_steps, atm_model, rho_w_sl, h_scale)

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
        fluorescence : Fluorescence
        """
        return sm.Fluorescence(self)

    def Cherenkov(self):
        """
        Calculate the Cherenkov light production from a shower profile
        discretization.

        Returns
        -------
        cherenkov : Cherenkov
        """
        return sm.Cherenkov(self)

    def show(self):
        """
        Show the shower profile, both number of charged particles and energy
        deposit, as a function of slant depth.

        Returns
        -------
        (ax1, ax2) : AxesSubplot
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


# Constructor #################################################################
def _profile(profile, E, theta, alt, prf_model, X_max, X0_GH, lambda_GH, zi,
             atmosphere, h0, h_top, N_steps, atm_model, rho_w_sl, h_scale):
    """
    Constructor of Profile class.

    Parameters
    ----------
    profile :  Profile
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
        proton showers is calculated from the radiation length
        lambda_r = 36.7 g/cm^2 and the critical energy E_c = 81 MeV.
    X0_GH : float
        X0 parameter in g/cm2 to be used when prf_model=='Gaisser-Hillas'.
        If None, a typical value for the input energy is used.
    lambda_GH : float
        Lambda parameter in g/cm2 to be used when prf_model=='Gaisser-Hillas'.
        If None, a typical value for the input energy is used.
    zi : float, default None
        Height in km of the first interaction point of the shower. 
    atmosphere : Atmosphere, default None
        If None, a new Atmosphere object is generated with parameters h0, h_top,
        N_steps, atm_model.
    h0 : float
        Ground level in km above sea level.
    h_top : float
        Top level of the atmosphere in km above sea level.
    N_steps : int
        Number of discretization steps.
    atm_model : int or DataFrame
        Atmospheric model assuming dry air.
    rho_w_sl : float
        Water-vapor density in g/cm^3 at sea level to calculate a simple
        exponential profile of water-vapor. Set to zero if dry air is assumed.
    h_scale : float
        Scale height in km to be used in the water-vapor exponential profile.
    """
    from .atmosphere import Atmosphere
    if isinstance(atmosphere, Atmosphere):
        pass
    elif atmosphere is None:
        atmosphere = sm.Atmosphere(h0, h_top, N_steps,
                                   atm_model, rho_w_sl, h_scale)
    else:
        raise ValueError('The input atmosphere is not valid.')

    # The columns of the output DataFrame are: the slant depth in g/cm^2,
    # the shower age s, the energy deposit E_dep in MeV, and the number N_ch of
    # charged particles with energy above 1MeV
    # profile = Profile(columns=['X', 's', 'dX', 'E_dep', 'N_ch'])
    profile.atmosphere = atmosphere

    # The input shower parameters along with some geometric parameters are
    # included as attributes of the DataFrame
    profile.E = E
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    profile.theta = theta
    profile.alt = alt
    if theta==180.:
        uz = -1.
    else:
        uz = np.cos(np.radians(theta))

    #Slant depth in g/cm^2 (following track.X_to_xyz())
    if zi is None:
        if uz>0.:
            Xv_i = 0.
        else:
            Xv_i = atmosphere.Xv_total
    else:
        Xv_i = atmosphere.h_to_Xv(zi + atmosphere.h0)

    X = (atmosphere.X_vert - Xv_i) / uz
    points = atmosphere[(X>0.) & (atmosphere.X_vert>0.)].index
    profile.X = X[points]

    # Length in km travelled through one atmospheric slice
    profile.dl = atmosphere.h_step / abs(uz)
    # Depth in g/cm^2 travelled through one atmospheric slice
    profile.dX = 100000. * profile.dl * atmosphere.rho[points]

    profile.prf_model = prf_model

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
        s = 3. * profile.X / (profile.X + 2. * X_max)  # Shower age
        profile.s = s

        profile.X0_GH = None
        profile.lambda_GH = None

        dE_dX = np.interp(profile.X, X_model, dE_dX_model, left=0., right=0.)
        profile.E_dep = dE_dX * profile.dX
        profile.N_ch = dE_dX / profile._alpha(profile.s)

        if E < profile.E_dep.sum():
            profile.E = profile.E_dep.sum()
            raise ValueError("""
                The input shower energy is lower than the integrated energy
                deposit profile.
                """)
        elif E > profile.E_dep.sum()*1.2:
            print("""
            Warning: The input energy is greater than the integrated energy
            deposit profile by more than 20%.
            """)

    elif prf_model == 'Greisen':
        if X_max is None:
            # E_c: critical energy in air in MeV
            # lambda_r: radiation length in air in g/cm2
            X_max = ct.lambda_r * np.log(E / ct.E_c)
        profile.X_max = X_max
        s = 3. * profile.X / (profile.X + 2. * X_max)  # Shower age
        profile.s = s

        profile.X0_GH = None
        profile.lambda_GH = None

        # Greisen profile: N_ch = 0.31/sqrt(t_max) * exp[ t * (1-3/2*log(s)) ]
        # where t=X/lambda_r ,
        N_ch = (0.31 / np.sqrt(X_max / ct.lambda_r)
                * np.exp(profile.X / ct.lambda_r * (1.-1.5*np.log(s))))

        # Shower size with an energy cut of 1MeV
        profile.N_ch = profile._Greisen_norm() * N_ch

        # Deposited energy in each slice: E_dep = alpha * N_ch * dX,
        # where  alpha(s) is the mean ionization energy loss per electron for
        # an energy cut of 1MeV
        profile.E_dep = profile._alpha(s) * profile.N_ch * profile.dX

    elif prf_model == 'Gaisser-Hillas':
        # If not given, a typical value according to Heitler model is used
        if X_max is None:
            # E_c: critical energy in air in MeV
            # lambda_r: radiation length in air in g/cm
            X_max = ct.lambda_r * np.log(E / 4. / ct.E_c)
        profile.X_max = X_max
        s = 3. * profile.X / (profile.X + 2. * X_max)  # Shower age
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
        N_ch = 0. * profile.X
        N_ch[X>X_min] = (
            ((X[X>X_min]-X0_GH) / (X_max-X0_GH))**((X_max-X0_GH)/lambda_GH)
            * np.exp(-(X[X>X_min]-X_max) / lambda_GH))
        # X>X_min prevents from errors for discretization steps with X<=0,
        # or with X<X0_GH, where the GH profile is undefined

        # Shower size with an energy cut of 1MeV
        profile.N_ch = profile._GH_norm() * N_ch

        # Deposited energy in each slice: E_dep = alpha * N_ch * dX,
        # where  alpha(s) is the mean ionization energy loss per electron for
        # an energy cut of 1MeV
        profile.E_dep = profile._alpha(profile.s) * profile.N_ch * profile.dX

    else:
        raise ValueError('The input model is not valid.')


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
    N_g = (0.31 / np.sqrt(X_max/ct.lambda_r) *
           np.exp(X/ct.lambda_r*(1.-1.5*np.log(s))))

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

    # Gaisser-Hillas profile:
    # N_ch = N_ch_max*[(X-X0)/(X_max-X0)]**((X_max-X0)/lambda)
    # *exp[-(X-X_max)/lambda]
    N_GH = np.zeros_like(X)
    N_GH[X>X0_GH] = (((X[X>X0_GH]-X0_GH)/(X_max-X0_GH))**((X_max-X0_GH)/lambda_GH)
                     *np.exp(-(X[X>X0_GH]-X_max)/lambda_GH))

    # Normalization constant K= E / int[alpha(s) * N_GH(s) * dX/ds *ds]
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

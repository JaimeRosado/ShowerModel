# coding: utf-8

import numpy as np
import pandas as pd
import showermodel.constants as ct

import warnings
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)

# Default values for Atmosphere
_Atmosphere__h0 = ct.config['Atmosphere']['h0']
_Atmosphere__h_top = ct.config['Atmosphere'].get('h_top') # optional parameter
_Atmosphere__N_steps = ct.config['Atmosphere']['N_steps']
_Atmosphere__atm_model = ct.config['Atmosphere']['atm_model']
_Atmosphere__rho_w_sl = ct.config['Atmosphere']['rho_w_sl']
_Atmosphere__h_scale = ct.config['Atmosphere']['h_scale']


# Class #######################################################################
class Atmosphere(pd.DataFrame):
    """
    DataFrame containing an atmosphere discretization.

    Use sm.Atmosphere() to construct the default Atmosphere object.

    Parameters
    ----------
    h0 : float, default 0.0
        Ground level in km above sea level.
    h_top : float or None, default None
        Upper limit in km above sea level of the atmosphere
        discretization. If None, the top level of the selected
        atmospheric model is taken. 
    N_steps : int, default 550
        Number of discretization steps.
    atm_model : int or DataFrame, default 1
        Atmospheric model assuming dry air. If an int value is given,
        atm_model is searched either from CORSIKA atmospheric models
        (from 1 to 29)
        or a file named atm_models.toml in the working directory
        containing user-defined models. If a DataFrame is given, it
        should have two columns, one labelled as h with height in km
        and other labelled as X_vert or P, depending on whether
        vertical depth in g/cm^2 or pressure in hPa is given.
    rho_w_sl : float, default 7.5e-6
        Water-vapor density in g/cm^3 at sea level to calculate a
        simple exponential profile of water-vapor. Set to zero if dry
        air is assumed.
    h_scale : float, default 2.0
        Scale height in km to be used in the water-vapor exponential
        profile.

    Attributes
    ----------
    h : float
        Column 0, height in km above sea level.
    X_vert : float
        Column 1, vertical depth in g/cm^2.
    rho : float
        Column 2, mass density in g/cm^3.
    temp : float
        Column 3, temperature in K.
    P : float
        Column 4, pressure in hPa.
    P_w : float
        Column 5, partial pressure of water vapor in hPa.
    E_th : float
        Column 6, Cherenkov energy threshold in MeV at 350 nm.
    r_M : float
        Column 7, Moliere radius in km.
    h0 : float
        Ground level in km above sea level.
    h_top : float
        Top level of the atmosphere in km above sea level.
    N_steps : int
        Number of discretization steps.
    h_step : float
        Size of discretization step in km.
    Xv_total : float
        Total vertical depth of the atmosphere.
    atm_model : int or DataFrame
        Atmospheric model assuming dry air.
    info : str
        Information about the atmospheric model. Set to df if
        atm_model is a DataFrame.
    rho_w_sl : float
        Water-vapor density in g/cm^3 at sea level.
    h_scale : float
        Scale height in km used in the water-vapor exponential profile.

    Methods
    -------
    h_to_Xv()
        Get vertical depth from height.
    h_to_rho()
        Get mass density from height.
    Xv_to_h()
        Get height from vertical depth.
    Xv_to_rho()
        Get density from vertical depth.
    Xv_to_P()
        Calculate pressure from vertical depth assuming constant acceleration
        of gravity.
    P_to_Xv()
        Calculate vertical depth from pressure assuming constant acceleration
        of gravity.
    Xv_rho_to_P_T()
        Calculate pressure and temperature from vertical depth and mass
        density assuming constant acceleration of gravity and an ideal gas.

    See also
    --------
    Track : DataFrame containing a shower track discretization.
    Profile : DataFrame containing a shower profile discretization.
    Shower : Make a discretization of a shower.
    """
    def __init__(self, h0=__h0, h_top=__h_top, N_steps=__N_steps,
                 atm_model=__atm_model, rho_w_sl=__rho_w_sl,
                 h_scale=__h_scale):
        super().__init__(
            columns=['h', 'X_vert', 'rho', 'temp', 'P', 'P_w', 'E_th', 'r_M'])
        _atmosphere(self, h0, h_top, N_steps, atm_model, rho_w_sl, h_scale)

    def h_to_Xv(self, h):
        """
        Get vertical depth in g/cm^2 from height in km above sea level.

        Parameters
        ----------
        h : float or array_like
            Height in km.

        Returns
        -------
        Xv : float or array_like
        """
        Xv, rho = self._h_to_Xv_rho(h)
        return Xv

    def h_to_rho(self, h):
        """
        Get mass density in g/cm^3 from height in km above sea level.

        Parameters
        ----------
        h : float or array_like
            Height in km.

        Returns
        -------
        rho : float or array_like
        """
        Xv, rho = self._h_to_Xv_rho(h)
        return rho

    def _h_to_Xv_rho(self, h):
        """
        Get both the vertical depth in g/cm^2 and the mass density in g/cm^3.
        from height in km above sea level.
        """
        if self._model is None:
            # 
            return _df_Xv_rho(h, self.atm_model)
        else:
            # _model contais the dictionary with atm_model
            return _model_Xv_rho(h, self._model)

    def Xv_to_h(self, Xv):
        """
        Get height in km above sea level from vertical depth in g/cm^2.

        Parameters
        ----------
        Xv : float or array_like
            Vertical depth in g/cm^2. If is outside the range of column
            X_vert, return None.

        Returns
        -------
        rho : float or array_like
        """
        h = np.append(self.h0, self.h)
        h = np.append(h, self.h_top)
        X_vert = np.append(self.Xv_total, self.X_vert)
        X_vert = np.append(X_vert, self.Xv_top)
        if self.Xv_top==0.:
            X_vert[-1] = X_vert[-2] / 1000.  # to avoid log(0)

        # An exponential atmosphere is assumed to interpolate h
        # The approximation is good enough for the top atmosphere too
        # (more or less linear)
        # X_vert should be ascending
        return 1.*np.interp(np.log(Xv), np.log(X_vert)[::-1], h[::-1],
                            left=None, right=None)

    def Xv_to_rho(self, Xv):
        """
        Get mass density in in g/cm^3 from vertical depth in g/cm^2.

        Parameters
        ----------
        Xv : float or array_like
            Vertical depth in g/cm^2. If is outside the range of column
            X_vert, return None.

        Returns
        -------
        rho : float or array_like
        """
        rho = np.append(self.rho0, self.rho)
        rho = np.append(rho, self.rho_top)
        X_vert = np.append(self.Xv_total, self.X_vert)
        X_vert = np.append(X_vert, self.Xv_top)

        # An exponential atmosphere is assumed so that rho has a linear
        # dependence on X_vert
        # X_vert should be ascending
        return 1.*np.interp(Xv, X_vert[::-1], rho[::-1], left=None, right=None)

    def Xv_to_P(self, Xv):
        """
        Calculate pressure from vertical depth assuming constant acceleration
        of gravity.

        Parameters
        ----------
        Xv : float or array_like
            Vertical depth in g/cm^2.

        Returns
        -------
        P : float or array_like
        """
        # g_cm: standard acceleration of gravity in cm/s^2
        return 1. * ct.g_cm * Xv / 1000. # hPa  (1 hPa = 1000 erg/cm^3)

    def P_to_Xv(self, P):
        """
        Calculate vertical depth from pressure assuming constant acceleration
        of gravity.

        Parameters
        ----------
        P : float or array_like
            Pressure in hPa.

        Returns
        -------
        P : float or array_like
        """
        # g_cm: standard acceleration of gravity in cm/s^2
        return 1. / ct.g_cm * P * 1000. # hPa  (1 hPa = 1000 erg/cm^3)

    def Xv_rho_to_P_T(self, Xv, rho):
        """
        Calculate pressure and temperature from vertical depth and mass
        density assuming constant acceleration of gravity and an ideal gas.
        """
        # M_air: air molar mass in g/mol (dry air)
        # R_erg: molar gas constant in erg/K/mol
        P = self.Xv_to_P(Xv)
        temp = np.zeros_like(rho)
        sel = rho>0.
        temp[sel] = (ct.M_air * ct.g_cm * self.X_vert[sel] / ct.R_erg /
                     self.rho[sel])

        return 1.*P, 1.*temp


# Constructor #################################################################
def _atmosphere(atmosphere, h0, h_top, N_steps, atm_model, rho_w_sl, h_scale):
    """
    Constructor of Atmosphere class.

    Parameters
    ----------
    atmosphere : Atmosphere object
    h0 : float
        Ground level in km above sea level.
    h_top : float or None
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
    # The output DataFrame includes the input parameters h0, h_top, N_steps,
    # model as attributes
    atmosphere.h0 = h0
    atmosphere.N_steps = N_steps
    atmosphere.atm_model = atm_model
    atmosphere.rho_w_sl = rho_w_sl
    atmosphere.h_scale = h_scale

    # Load atmospheric model
    if isinstance(atm_model, pd.DataFrame): # User-defined model
        # Data ordered by h to allow for interpolation
        atm_model = atm_model.sort_values(by='h', axis=0, ascending=True)
        atmosphere._model = None  # to be used in _h_to_Xv_rho
        atmosphere.info = 'DataFrame'
        
        h_top_model = atm_model.h.iloc[-1]
        if h_top is None:
            h_top = h_top_model
        elif h_top>h_top_model:
            h_top = h_top_model
        
        # Array of mid heights of the discretization of the atmosphere.
        # h0 represents the ground level.
        h = np.linspace(h0, h_top, N_steps+1)
        h_step = h[1] - h[0]
        h = h - h_step/2.
        h[0] = h0  # keep h0 to obtain the total vertical depth Xv_total
        h = np.append(h, h_top)  # keep h_top to obtain Xv_top
        
        # Vertical depth in g/cm2 and density in g/cm3 for h from DataFrame
        X_vert, rho = _df_Xv_rho(h, atm_model)
        
    else: # Get atmospheric parameters of the selected model from atm_models
        model = ct.atm_models.get(str(atm_model))
        if model is None:
            raise ValueError('This atm_model is not implemented.')
        atmosphere._model = model
        atmosphere.info = model.get('info')

        h_top_model = _model_h_top(model)
        if h_top is None:
            h_top = h_top_model
        elif h_top>h_top_model:
            h_top = h_top_model

        # Array of mid heights of the discretization of the atmosphere.
        # h0 represents the ground level.
        h = np.linspace(h0, h_top, N_steps+1)
        h_step = h[1] - h[0]
        h = h - h_step/2.
        h[0] = h0  # keep h0 to obtain the total vertical depth Xv_total
        h = np.append(h, h_top)  # keep h_top to obtain Xv_top

        # Vertical depth in g/cm2 and density in g/cm3 for h from the selected model
        X_vert, rho = _model_Xv_rho(h, model)

    atmosphere.h_top = h_top
    atmosphere.h_step = h_step
    atmosphere.Xv_total = X_vert[0]
    atmosphere.Xv_top = X_vert[-1]  # usually Xv_top=0
    atmosphere.rho_0 = rho[0]
    atmosphere.rho_top = rho[-1]
    h = h[1:-1]  # only mid heights are stored in columns
    X_vert = X_vert[1:-1]
    rho = rho[1:-1]
    atmosphere.h = h
    atmosphere.X_vert = X_vert
    atmosphere.rho = rho
    
    # It is assumed that air is an ideal gas and that the acceleration of
    # gravity is constant
    P, temp = atmosphere.Xv_rho_to_P_T(X_vert, rho)
    atmosphere.P = P
    atmosphere.temp = temp
        
    if rho_w_sl>0.:
        # R_erg: molar gas constant in erg/K/mol
        # M_w: water molar mass in g/mol
        # Water-vapor density in g/cm3
        rho_w = rho_w_sl * np.exp(-h / h_scale)
        P_w = ct.R_erg / ct.M_w * rho_w * temp / 1000. # hPa  (1 hPa = 1000 erg/cm^3)
        P_w_min = 2.e-6 * P
        P_w[P_w<P_w_min] = P_w_min[P_w<P_w_min]
    else:
        P_w = np.zeros_like(temp)
    atmosphere.P_w = P_w

    # delta=1-n at 350nm. J.C. Owens, Appl. Opt. 6 (1967) 51
    # P (calculated for dry air) is assumed to be the total pressure
    delta = np.zeros_like(temp)
    sel = temp>0.
    pw = P_w[sel]
    p = P[sel] - pw
    t = temp[sel]
    delta[sel] = 0.00000001 * (
        8132.8589 * p / t * (1. + p *
            (0.000000579 - 0.0009325 / t + 0.25844 / t**2)) +
        6961.9879 * pw / t * (1. + pw * (1. + 0.00037 * pw) *
            (-0.00237321 + 2.23366 / t - 710.792 / t**2 + 0.000775141 / t**3)))

    # Threshold energy for Cherenkov production at 350 nm in air
    # mc2: electron mass in MeV
    E_th = np.zeros_like(delta)
    E_th[sel] = ct.mc2 / np.sqrt(2.*delta[sel])
    atmosphere.E_th = E_th

    # Moliere radius
    # E_c: critical energy in air in MeV
    # lambda_r: radiation length in air in g/cm2
    r_M = np.zeros_like(rho)
    r_M[sel] = (4.*ct.pi/ct.alpha)**0.5 * (ct.mc2 / ct.E_c * ct.lambda_r
                                           / rho[sel] * 1.e-5)
    atmosphere.r_M = r_M

# Auxiliary functions #########################################################
def _model_Xv_rho(h, model):
    """
    Get vertical depth in g/cm^2 and mass density in g/cm^3 from height in km
    above sea level from an atmospheric model.

    Parameters
    ----------
    h : float or array_like
        Height in km.
    model : dict
        CORSIKA atmospheric model. Presently either 1 or 17. More models to
        be implemented.

    Returns
    -------
    Xv : float or array_like
        Vertical depth in g/cm^2.
    rho : float or array_like
        Mass density in g/cm^3.
    """
    h = np.array(h)
    Xv = np.zeros_like(h)
    rho = np.zeros_like(h)
    
    # For the atmospheric models used in CORSIKA, all the atmospheric
    # layers are exponential except for the top layer, which is linear
    # First, take all the layers, except the last one
    layers = zip(model['h_low'][:-1], # h_low in km
                 model['h_low'][1:],  # h_up in km
                 model['a'][:-1],     # g/cm2
                 model['b'][:-1],     # g/cm2
                 model['c'][:-1])     # cm

    # Exponential layers
    for h_low, h_up, a, b, c in layers:
        layer = (h>=h_low) & (h<h_up)
        Xv[layer] = a + b * np.exp(-h[layer] * 1.e5 / c)
        rho[layer] = (Xv[layer] - a) / c

    # Top layer (lineal)
    h_low = model['h_low'][-1]
    a = model['a'][-1]
    b = model['b'][-1]
    c = model['c'][-1]
    rho_up = b / c
    h_up = a / rho_up / 1.e5
    layer = (h>=h_low) & (h<h_up)
    Xv[layer] = a - rho_up * h[layer] * 1.e5
    # Constant density
    rho[layer] = rho_up

    # If the input h is a scalar, then the function returns two scalars
    return 1.*Xv, 1.*rho


def _df_Xv_rho(h, df):
    """
    Interpolate the vertical depth in g/cm^2 from a DataFrame and calculate
    the mass density in g/cm^3.

    Parameters
    ----------
    h : float or array_like
        Height in km.
    df : DataFrame
        DataFrame containing height and vertical depth.

    Returns
    -------
    Xv : float or array_like
        Vertical depth in g/cm^2.
    rho : float or array_like
        Mass density in g/cm^3.
    """
    # Check lower limit
    if h[0]<df.h.iloc[0]:
        raise ValueError(
            'The input DataFrame should include the ground level h0.')
    h_df = np.array(df.h)

    Xv_df = df.get('X_vert')
    if Xv_df is None:
        P = df.get('P')
        if P is None:
            raise ValueError(
                'The input DataFrame should have a column X_vert or P.')
        else:
            Xv_df = 1. / ct.g_cm * P * 1000.
    Xv_df = np.array(Xv_df)

    # Check upper limit
    if Xv_df[-1]==0.:
        Xv_df[-1] = Xv_df[-2] / 1000. # to avoid log(0)

    # An exponential atmosphere is assumed for interpolation and
    # numerical derivative
    log_Xv_df = np.log(Xv_df)
    log_Xv_lim = log_Xv_df[-1]

    # Vertical depth. Xv=0 for h>h_df[-1]
    Xv = np.exp(np.interp(h, h_df, log_Xv_df, right=-np.inf))

    # Mass density. Expected to be constant at the top
    rho_df = -Xv_df * (np.diff(log_Xv_df, append=0.) /
                       np.diff(h_df, append=0.) * 1.e-5)
    rho_df[-1] = rho_df[-2]
    # Xv_df and rho_df must be ascending, but both Xv and rho are descending
    # rho=0 for Xv=0 (h>h_df[-1])
    rho = np.interp(Xv, Xv_df[::-1], rho_df[::-1], left=0.)

    # If the input h is a scalar, then the function returns Xv and rho as scalars
    return 1.*Xv, 1.*rho

def _model_h_top(model):
    """
    Get the top height of the atmosphere.
    """
    # Top layer (lineal)
    a = model['a'][-1]
    b = model['b'][-1]
    c = model['c'][-1]
    return a / b * c / 1.e5


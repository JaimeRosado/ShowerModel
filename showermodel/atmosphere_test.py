# coding: utf-8

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)

# Default values for atmosphere
_h0 = 2.20  # km
_h_top = 112.8292  # km
_N_steps = 550
_model = 1


# Constructor #################################################################
def Atmosphere(h0=_h0, h_top=_h_top, N_steps=_N_steps, model=_model):
    """
    Make an atmosphere discretization.

    Atmosphere() makes the default Atmosphere object.

    Parameters
    ----------
    h0 : float
        Ground level in km above sea level.
    h_top : float
        Top level of the atmosphere in km above sea level.
    N_steps : int
        Number of discretization steps.
    model : int
        CORSIKA atmospheric model. Presently either 1 or 17. More models to
        be implemented.

    Returns
    -------
    atmosphere : Atmosphere object.

    See also
    --------
    _Atmosphere : Atmosphere class.
    Track : Constructor of Track object.
    Profile: Constructor of Profile object.
    Shower : Constructor of Shower object.
    """
    atmosphere = _Atmosphere(
        columns=['h', 'X_vert', 'rho', 'temp', 'P', 'P_w', 'E_th', 'r_M'])

    # For the default atmospheric parameters
    if (h0 == _h0) and (N_steps == _N_steps) and (model == _model):
        global ATM
        try:
            return ATM  # Outputs the default atmosphere if already generated
        except Exception:
            # If the default atmosphere is to be generated,
            # it will be assigned to the global variable ATM
            ATM = atmosphere

    # The output DataFrame includes the input parameters h0, h_top, N_steps,
    # model as attributes
    atmosphere.h0 = h0
    atmosphere.h_top = h_top
    atmosphere.N_steps = N_steps
    atmosphere.model = model

    # Array of mid heights of the discretization of the atmosphere.
    # h0 represent the ground level.
    height = np.linspace(h0, atmosphere.h_top, N_steps+1)
    atmosphere.h_step = height[1] - height[0]
    atmosphere.h = height[1:] - atmosphere.h_step/2.

    # Vertical depth and density from the input model
    Xv, rho = atmosphere._get_Xv_rho(atmosphere.h)

    atmosphere.X_vert = Xv
    atmosphere.rho = rho

    temp = np.zeros_like(rho)
    # Air is assumed to be an ideal gas with 28.96 g/mol
    temp[rho > 0] = 28.96 * 9.81 * Xv[rho > 0] / 831445.98 / rho[rho > 0]
    atmosphere.temp = temp
    P = 9.81 * Xv / 10.  # A constant gravitational acceleration is assumed
    atmosphere.P = P
    # CORSIKA models do not describe the partial pressure of water vapor
    P_w = np.zeros_like(P)
    atmosphere.P_w = P_w

    # delta=1-n at 350nm. J.C. Owens, Appl. Opt. 6 (1967) 51
    delta = np.zeros_like(temp)
    delta[temp > 0] = 0.00000001 * (
        8132.8589*(P[temp > 0]-P_w[temp > 0])/temp[temp > 0]
        * (1.+(P[temp > 0]-P_w[temp > 0])
            * (0.000000579-0.0009325/temp[temp > 0]+0.25844/temp[temp > 0]**2))
        + 6961.9879*P_w[temp > 0]/temp[temp > 0]
        * (1.+P_w[temp > 0]*(1.+0.00037*P_w[temp > 0])
            * (-0.00237321+2.23366/temp[temp > 0]-710.792/temp[temp > 0]**2
                + 0.000775141/temp[temp > 0]**3)))

    # Threshold energy for Cherenkov production at 350 nm in air
    atmosphere.E_th = 0.511 / np.sqrt(2.*delta)

    # Moliere radius in km
    atmosphere.r_M = 21.2 / 81. * 36.7 / atmosphere.rho / 100000.

    return atmosphere


# Class #######################################################################
class _Atmosphere(pd.DataFrame):
    """
    DataFrame containing an atmosphere discretization.

    Use sm.Atmosphere to construct an Atmosphere object.

    Columns
    -------
    h : float
        Height in km above sea level.
    X_vert : float
        Vertical depth in g/cm^2.
    rho : float
        Mass density in g/cm^3.
    temp : float
        Temperature in K.
    P : float
        Pressure in hPa.
    P_w : float
        Partial pressure of water vapor in hPa.
    E_th : float
        Cherenkov energy threshold in MeV at 350 nm.
    r_M : float
        Moliere radius in km.

    Attributes
    ----------
    h0 : float
        Ground level in km above sea level.
    h_top : float
        Top level of the atmosphere in km above sea level.
    N_steps : int
        Number of discretization steps.
    h_step : float
        Size of discretization step in km.
    model : int
        CORSIKA atmospheric model. Presently either 1 or 17. More models to
        be implemented.

    Methods
    -------
    h_to_Xv : Get vertical depth from height.
    h_to_rho : Get mass density from height.
    Xv_to_h : Get height from vertical depth.

    See also
    --------
    Atmosphere : Constructor of Atmosphere object.
    Track : Constructor of Track object.
    Profile : Constructor of Profile object.
    Shower : Constructor of Shower object.
    """

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
            Vertical depth in g/cm^2.
        """
        Xv, rho = self._get_Xv_rho(h)
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
            Mass density in g/cm^3.
        """
        Xv, rho = self._get_Xv_rho(h)
        return rho

    def _get_Xv_rho(self, h):
        """
        Get both the vertical depth in g/cm^2 and the mass density in g/cm^3
        from height in km above sea level.
        """
        return _get_Xv_rho(h, self.model)

    def Xv_to_h(self, Xv):
        """
        Get height in km above sea level from vertical depth in g/cm^2.

        Parameters
        ----------
        Xv : float or array_like
            Vertical depth in g/cm^2.

        Returns
        -------
        h : float or array_like
            Height in km.
        """
        if Xv == 0:
            return self.h_top

        h_lower = self.h[self.X_vert > Xv].max()  # Lower bound for h
        h_upper = self.h[self.X_vert < Xv].min()  # Lower bound for h
        # Upper and lower bounds for Xv (corresponding to h_lower and  h_upper)
        Xv_upper = self.X_vert[self.X_vert > Xv].min()
        Xv_lower = self.X_vert[self.X_vert < Xv].max()
        # An exponential atmosphere is assumed to interpolate h
        # The approximation is good enough for the top atmosphere too
        # (more or less linear)
        return (h_lower + (h_upper-h_lower) / np.log(Xv_upper/Xv_lower)
                * np.log(Xv_upper/Xv))


# Auxiliary functions #########################################################
def _get_Xv_rho(h, model):
    """
    Get vertical depth in g/cm^2 and mass density in g/cm^3 from height in km
    above sea level for an atmospheric model.

    Parameters
    ----------
    h : float or array_like
        Height in km.
    model : int
        CORSIKA atmospheric model. Presently either 1 or 17. More models to
        be implemented.

    Returns
    -------
    Xv : float or array_like
        Vertical depth in g/cm^2.
    rho : float or array_like
        Mass density in g/cm^3.
    """
    # Some atmospheric models used in CORSIKA.
    # Each row contains the model parameters for a different atmospheric layer:
    # h_ini(km), a(g/cm^2), b(g/cm^2), c(km)
    if model == 1:  # model 1
        h_top = 112.8292  # Height of the atmosphere at which Xv=0
        #     h_ini(km),     a(g/cm^2), b(g/cm^2),          c(km)
        param = ((100.0,    0.01128292,    1.00000, 10000.0000000),
                 (040.0,    0.00000000,  540.17780,     7.7217016),
                 (010.0,    0.61289000, 1305.59480,     6.3614304),
                 (004.0,  -94.91900000, 1144.90690,     8.7815355),
                 (000.0, -186.55530600, 1222.65620,     9.9418638))
    elif model == 17:
        h_top = 112.8292
        #     h_ini(km),     a(g/cm^2),  b(g/cm^2),         c(km)
        param = ((100.0,    0.01128292,    1.00000, 10000.0000000),
                 (037.0,    0.00043545,  655.67307,     7.3752177),
                 (011.4,    0.61289000, 1322.97480,     6.2956893),
                 (007.0,  -57.93248600, 1143.04250,     8.0000534),
                 (000.0, -149.80166300, 1183.60710,     9.5424834))
    else:
        raise ValueError('This input model is not implemented yet')

    h = np.array(h)
    Xv = np.zeros_like(h)
    rho = np.zeros_like(h)

    # For the atmospheric models used in CORSIKA, the different atmospheric
    # layers are exponential except for the top layer, which is linear
    for (i, (h_ini, a, b, c)) in enumerate(param):
        if i == 0:                  # Top layer
            h_fin = h_top           # h_top is such that Xv=0 for h=h_top
            Xv[(h >= h_ini) & (h < h_fin)] = (
                a - b * h[(h >= h_ini) & (h < h_fin)] / c)
            # Constant density
            rho[(h >= h_ini) & (h < h_fin)] = b / c / 100000.
        else:  # Exponential model for the other layers
            Xv[(h >= h_ini) & (h < h_fin)] = (
                a + b * np.exp(-h[(h >= h_ini) & (h < h_fin)] / c))
            rho[(h >= h_ini) & (h < h_fin)] = (
                (Xv[(h >= h_ini) & (h < h_fin)] - a) / c / 100000.)
        h_fin = h_ini

    # If the input h is a scalar, then the function returns two scalars
    return 1.*Xv, 1.*rho

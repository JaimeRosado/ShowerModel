# coding: utf-8

import pandas as pd
import warnings
import matplotlib.pyplot as plt
import showermodel.constants as ct

warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)

# Default values for Fluorescence
_Fluorescence__wvl_ini = ct.config['Signal']['wvl_ini']
_Fluorescence__wvl_fin = ct.config['Signal']['wvl_fin']

# Class #######################################################################
class Fluorescence(pd.DataFrame):
    """
    DataFrame containing the fluorescence light production.

    Fluorescence light is evaluated at each of the 57 bands of the fluorescence
    spectrum in the 280 - 670 nm range based on the parameterization described
    in D. Morcuende et al., Astropart. Phys. 107(2019)26 and references therein.

    Parameters
    ----------
    profile : Profile, mandatory
        Profile object to be used.

    Attributes
    ----------
    281 : float
        Column 0, number of fluorescence photons in the band centered at 281 nm.
    282 : float
        Column 1, number of fluorescence photons in the band centered at 282 nm.
    296 : float
        Column 2, number of fluorescence photons in the band centered at 296 nm.
    ... :float
        Column ...
    666 : float
        Column 56, number of fluorescence photons in the band centered at 666 nm.
    profile : Profile
    atmosphere : Atmosphere

    Methods
    -------
    show()
        Show the production of fluorescence photons in the 290 - 430 nm range
        as a function of slant depth.
    """
    
    def __init__(self, profile):
        # All bands, including those outside the wavelength interval
        super().__init__(columns=ct.fluo_model['wvl'])
        _fluorescence(self, profile)

    def show(self):
        """
        Show the production of fluorescence photons in the 290 - 430 nm range
        as a function of slant depth.

        Returns
        -------
        ax : AxesSubplot
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        # Selection of bands within the wavelength interval
        fluo = self.loc[:, __wvl_ini:__wvl_fin]
        ax.plot(self.profile.X, fluo.sum(axis=1), 'b-')
        ax.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax.axes.yaxis.set_label_text(
            "Fluorescence production (Photons·cm$^2$/g)")
        return ax


# Constructor #################################################################
def _fluorescence(fluorescence, profile):
    """
    Constructor of Fluorescence class.

    Parameters
    ----------
    fluorescence : Fluorescence
    profile : Profile
    """
    # Load fluorescence parameters
    Y0_337 = ct.fluo_model['Y0_337']
    P0 = ct.fluo_model['P0']
    T0 = ct.fluo_model['T0']
    bands = zip(ct.fluo_model['wvl'],
                ct.fluo_model['I_rel'],
                ct.fluo_model['PP0'],
                ct.fluo_model['PPw'],
                ct.fluo_model['a'])
    
    # Number of fluorescence photons at 337nm
    N0_337 = Y0_337 * profile.E_dep

    # fluorescence = Fluorescence()
    fluorescence.profile = profile
    fluorescence.atmosphere = profile.atmosphere

    # Only discretization steps where the profile is defined
    atm = profile.atmosphere.iloc[profile.index]
    P = atm.P - atm.P_w
    temp = atm.temp
    P_w = atm.P_w

    # Number of emitted photons at wavelength wvl as a function of pressure P,
    # temperature T and partial pressure P_w of water vapor:
    for wvl, I_rel, PP0, PPw, a in bands:
        # For some bands, no information on the quenching contribution from
        # water vapor is available
        if PPw == 0:
            P_PP = P / PP0
        else:
            P_PP = P / PP0 + P_w / PPw

        fluorescence[wvl] = (
            N0_337 * I_rel * (1. + P0 / PP0) / (1. + P_PP
                                                * (T0 / temp)**(0.5 - a)))

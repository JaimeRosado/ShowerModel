# coding: utf-8

import pandas as pd
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)


# Constructor #################################################################
def Fluorescence(profile):
    """
    Calculate the fluorescence photon production from a shower profile.

    The parameterization described in D. Morcuende et al., Astropart. Phys.
    107(2019)26 and references therein is used.

    Parameters
    profile : Profile object.

    Returns
    -------
    fluorescence : Fluorescence object.

    See also
    --------
    _Fluorescence : Fluorescence class.
    """
    # Parameters of the fluorescence model (34 bands)
    #     wvl(nm),   Irel,   PP0,  PPw,     a
    model = ((296, 0.0516, 18.50, 0.00,  0.00),
             (298, 0.0277, 17.30, 0.00,  0.00),
             (302, 0.0041, 21.00, 0.00,  0.00),
             (308, 0.0144, 21.00, 0.00,  0.00),
             (312, 0.0724, 18.70, 0.00,  0.00),
             (314, 0.1105, 12.27, 1.20, -0.13),
             (316, 0.3933, 11.88, 1.10, -0.19),
             (318, 0.0046, 21.00, 0.00,  0.00),
             (327, 0.0080, 19.00, 0.00,  0.00),
             (329, 0.0380, 20.70, 0.00,  0.00),
             (331, 0.0215, 16.90, 0.00,  0.00),
             (334, 0.0402, 15.50, 0.00,  0.00),
             (337, 1.0000, 15.89, 1.28, -0.35),
             (346, 0.0174, 21.00, 0.00,  0.00),
             (350, 0.0279, 15.20, 1.50, -0.38),
             (354, 0.2135, 12.70, 1.27, -0.22),
             (358, 0.6741, 15.39, 1.30, -0.35),
             (366, 0.0113, 21.00, 0.00,  0.00),
             (367, 0.0054, 19.00, 0.00,  0.00),
             (371, 0.0497, 14.80, 1.30, -0.24),
             (376, 0.1787, 12.82, 1.10, -0.17),
             (381, 0.2720, 16.51, 1.40, -0.34),
             (386, 0.0050, 19.00, 0.00,  0.00),
             (388, 0.0117,  7.60, 0.00,  0.00),
             (389, 0.0083,  3.90, 0.00,  0.00),
             (391, 0.2800,  2.94, 0.33, -0.79),
             (394, 0.0336, 13.70, 1.20, -0.20),
             (400, 0.0838, 13.60, 1.10, -0.20),
             (405, 0.0807, 17.80, 1.50, -0.37),
             (414, 0.0049, 19.00, 0.00,  0.00),
             (420, 0.0175, 13.80, 0.00,  0.00),
             (424, 0.0104,  3.90, 0.00,  0.00),
             (427, 0.0708,  6.38, 0.00,  0.00),
             (428, 0.0494,  2.89, 0.60, -0.54))
    # Reference atmospheric conditions
    P0 = 800.
    T0 = 293.
    # Y0_337=7.05 ph/MeV is the fluorescence yield of the band at 337nm in
    # dry air at reference conditions
    # J. Rosado et al. Astropart. Phys. 55(2014)51.
    N0_337 = 7.05 * profile.E_dep

    fluorescence = _Fluorescence()
    fluorescence.profile = profile
    fluorescence.atmosphere = profile.atmosphere

    # Number of emitted photons at wavelength wvl as a function of pressure P,
    # temperature T and partial pressure P_w of water vapor:
    for i, (wvl, I_rel, PP0, PPw, a) in enumerate(model):
        # For some bands, no information on the quenching contribution from
        # water vapor is available
        if PPw == 0:
            fluorescence[wvl] = (
                N0_337 * I_rel * (1. + P0 / PP0)
                / (1. + profile.atmosphere.P / PP0
                   * (T0 / profile.atmosphere.temp)**(0.5 - a)))
        else:
            fluorescence[wvl] = (
                N0_337 * I_rel * (1. + P0 / PP0 / (
                    1. + ((profile.atmosphere.P - profile.atmosphere.P_w)
                          / PP0 + profile.atmosphere.P_w / PPw)
                    * (T0 / profile.atmosphere.temp)**(0.5 - a))))

    return fluorescence


# Class #######################################################################
class _Fluorescence(pd.DataFrame):
    """
    DataFrame containing the fluorescence light production at each of the
    34 bands of the fluorescence spectrum in the 290 - 430 nm range based on
    the parameterization described in D. Morcuende et al.,
    Astropart. Phys. 107(2019)26 and references therein.

    Columns
    -------
    296 : float
        Number of fluorescence photons in the band centered at 296 nm.
    ...
    428 : float
        Number of fluorescence photons in the band centered at 428 nm.

    Atributes
    ---------
    profile : Profile object.
    atmosphere : Atmosphere object.

    Methods
    -------
    show : Show the production of fluorescence photons in the 290 - 430 nm
        range as a function of slant depth.
    """
    def show(self):
        """
        Show the production of fluorescence photons in the 290 - 430 nm range
        as a function of slant depth.

        Returns
        -------
        ax : AxesSubplot object.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.plot(self.profile.X, self.sum(axis=1), 'b-')
        ax.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax.axes.yaxis.set_label_text(
            "Fluorescence production (Photons·cm$^2$/g)")
        return ax

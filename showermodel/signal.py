# coding: utf-8

import numpy as np
import pandas as pd
import showermodel as sm
import matplotlib.pyplot as plt


# Constructor #################################################################
def Signal(telescope, shower, projection=None, atm_trans=True, tel_eff=True,
           **kwargs):
    """
    Calculate the signal produced by a shower detected by a telescope.

    Parameters
    ----------
    telescope : Telescope object.
    shower : Shower object.
    projection : Projection object
        If None, it will generated from telescope and shower.
    atm_trans : bool, default True
        Include the atmospheric transmision to transport photons.
    tel_eff : bool, default True
        Include the telescope efficiency to calculate the signal. If False,
        100% efficiency is assumed for a given wavelenght interval.
    **kwargs {wvl_ini, wvl_fin, wvl_step}
        These parameters will modify the wavelenght interval when
        tel_eff==False. If None, the wavelength interval defined in the
        telescope is used.

    Returns
    -------
    signal : Signal object.
    """
    from .telescope import _Telescope
    from .shower import _Shower
    if not isinstance(telescope, _Telescope):
        if not isinstance(telescope, _Shower):
            raise ValueError('The input telescope is not valid')
        else:
            telescope, shower = (shower, telescope)
    if not isinstance(shower, _Shower):
        raise ValueError('The input shower is not valid')

    # This function is normally called from Event. If not, projection must be
    # generated.
    from .projection import _Projection
    if not isinstance(projection, _Projection):
        projection = sm.Projection(telescope, shower.track)
    atmosphere = shower.atmosphere
    track = shower.track
    fluorescence = shower.fluorescence
    cherenkov = shower.cherenkov

    signal = _Signal()
    signal.shower = shower
    signal.telescope = telescope
    signal.projection = projection
    signal.atmosphere = atmosphere
    signal.track = track
    signal.profile = shower.profile
    signal.fluorescence = fluorescence
    signal.cherenkov = cherenkov

    signal.atm_trans = atm_trans
    signal.tel_eff = tel_eff

    if tel_eff:
        # Wavelenght range to calculate the signal
        wvl_ini = telescope.wvl_ini
        wvl_fin = telescope.wvl_fin
        wvl_step = telescope.wvl_step
        wvl_cher = telescope.wvl_cher
        eff_fluo = telescope.eff_fluo
        eff_cher = telescope.eff_cher
    else:
        # User-defined wavalength range
        wvl_ini = kwargs.get('wvl_ini', telescope.wvl_ini)
        wvl_fin = kwargs.get('wvl_fin', telescope.wvl_fin)
        wvl_step = kwargs.get('wvl_step', telescope.wvl_step)
        wvl_cher = np.arange(wvl_ini, wvl_fin, wvl_step)
    signal.wvl_ini = wvl_ini
    signal.wvl_fin = wvl_fin
    signal.wvl_step = wvl_step

    # Only discretization points within the telescope field of view contributes
    # to the signal. In addition, the very begninning of the shower profile is
    # ignored to speed up calculations
    points = projection[projection.FoV & (signal.profile.s > 0.01)].index
    distance = np.array(projection.distance.loc[points])
    theta = np.radians(projection.theta.loc[points])
    alt = np.radians(projection.alt.loc[points])

    # Solid angle fraction covered by the telescope area. Only discretization
    # points within the telescope field of view contributes to the signal
    collection = (telescope.area * np.cos(theta) / 4000000. / np.pi
                  / distance**2)

    # Collection efficiency for the angular distribution of Cherenkov light
    # See F. Nerling et al., Astropart. Phys. 24(2006)241.
    beta = np.radians(projection.beta.loc[points])
    theta_c = np.radians(cherenkov.theta_c.loc[points])
    theta_cc = np.radians(cherenkov.theta_cc.loc[points])
    a = np.array(cherenkov.a.loc[points])
    b = np.array(cherenkov.b.loc[points])
    collection_cher = collection * 2. / np.sin(beta) * (
        a / theta_c * np.exp(-beta / theta_c)
        + b / theta_cc * np.exp(-beta / theta_cc))

    # Relative fluorescence contribution from each shower point at each band
    # (between wvl_ini and wvl_fin). The atmospheric transmission is included
    # later
    rel_fluo = fluorescence.loc[points]
    if tel_eff:
        rel_fluo *= eff_fluo  # 34 bands
    # Selection of bands within the wavelenght range
    rel_fluo = rel_fluo.loc[:, wvl_ini:wvl_fin]

    if atm_trans:
        # Atmospheric transmission at 350 nm. Only Rayleigh scattering is
        # considered
        X_vert = np.array(atmosphere.X_vert.loc[points])
        rho = np.array(atmosphere.rho.loc[points])
        thickness = np.array(atmosphere.h_to_Xv(atmosphere.h0 + telescope.z)
                             - X_vert)
        thickness[thickness != 0] = (thickness[thickness != 0]
                                     / np.sin(alt[thickness != 0]))
        thickness[thickness == 0] = (100000. * distance[thickness == 0]
                                     * rho[thickness == 0])
        # Only points within the telescope FoV, otherwise trans=0
        trans = np.exp(-thickness / 1645.)

        # Relative fluorescence contribution including atmospheric transmission
        for wvl in rel_fluo:
            rel_fluo[wvl] *= trans ** ((350. / wvl)**4)

        # Wavelenght factor for Cherenkov contribution to signal from each
        # shower point
        wvl_factor = pd.DataFrame(index=points)
        for wvl in wvl_cher:
            wvl_factor[wvl] = trans ** ((350. / wvl)**4) / wvl**2
            # wvl**2 -> (wvl**2 - wvl_step**2 / 4.)
        if tel_eff:
            wvl_factor *= eff_cher
        wvl_factor = wvl_factor.sum(axis=1) * wvl_step / (1./290.-1./430.)

    elif tel_eff:  # If atmospheric transmission is not included
        # The wavelength factor of Cherenkov signal is the same for all
        # shower points
        wvl_factor = eff_cher / wvl_cher**2
        # wvl_cher**2 -> (wvl_cher**2 - wvl_step**2 / 4.)
        wvl_factor = wvl_factor.sum() * wvl_step / (1./290.-1./430.)

    # If neither the atmospheric transmission nor the telescope efficiency are
    # included
    else:
        # The wavelength factor of Cherenkov signal only depends on the
        # integration wavelength interval
        wvl_factor = (1. / wvl_ini - 1. / wvl_fin) / (1. / 290. - 1. / 430.)

    # Number of photoelectrons due to fluorescence light emitted from each
    # shower point
    signal['Npe_fluo'] = rel_fluo.sum(axis=1) * collection
    # Number of photoelectrons due to fluorescence light within the FoV
    signal.Npe_fluo_sum = signal.Npe_fluo.sum()
    # Number of photoelectrons due to Cherenkov light emitted from each shower
    # point
    signal['Npe_cher'] = (cherenkov.N_ph.loc[points] * collection_cher
                          * wvl_factor)
    # Number of photoelectrons due to Cherenkov light within the FoV
    signal.Npe_cher_sum = signal.Npe_cher.sum()
    # Total number of photoelectrons from both light components emitted at each
    # shower point
    signal['Npe_total'] = signal.sum(axis=1)
    # Total number of photoelectrons
    signal.Npe_total_sum = signal.Npe_cher_sum + signal.Npe_fluo_sum

    return signal


# Class #######################################################################
class _Signal(pd.DataFrame):
    """
    DataFrame containing the signal produced by a shower detected by a
    telescope.

    Columns
    -------
    Npe_cher : float
        Number of photoelectrons per discretization step due to
        Cherenkov light.
    Npe_fluo : float
        Number of photoelectrons per discretization step due to
        fluorescence light.
    Npe_total : float
        Total number of photoelectrons per discretization step.

    Attributes
    ----------
    telescope : Telescope object.
    atm_trans : bool
        True if the atmospheric transmision is included.
    tel_eff : bool
        True if the telescope efficiency is included.
    wvl_ini : float
        Initial wavelength in nm of the interval.
    wvl_fin : float
        Final wavelength in nm of the interval.
    wvl_step : float
        Step size in nm of the interval.
    Npe_cher_sum : float
        Sum of photoelectrons due to Cherenkov light.
    Npe_fluo_sum : float
        Sum of photoelectrons due to fluorescence light.
    Npe_total_sum : float
        Sum of photoelectrons due to both light components.
    shower : Shower object.
    projection : Projection object.
    fluorescence : Fluorescence object.
    cherenkov : Cherenkov object.
    profile : Profile object.
    track : Track object.
    atmosphere : Atmosphere object.

    Methods
    -------
    show_projection : Show the projection of the shower track viewed by the
        telescope in both zenith and camera projections.
    show_profile : Show the shower profile, both number of charged particles
        and energy deposit, as a function of slant depth.
    show_light_production : Show the production of both Cherenkov and
        fluorescence photons in the 290 - 430 nm range as a function of slant
        depth.
    show : Show the signal evolution as a function of both time and beta angle
        (relative to the shower axis direction).
    Image : Generate a time-varying shower image.
    """

    def show_projection(self, shower_size=True, axes=True, max_theta=30.,
                        X_mark='X_max'):
        """
        Show the projection of the shower track viewed by the telescope in both
        horizontal and FoV coordinates systems.

        Parameters
        ----------
        shower_size : bool, default True
            Make the radii of the shower track points proportional to
            the shower size.
        axes : bool, default True
            Show the axes of both frames of reference.
        max_theta : float, default 30 degrees
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure, default to X_max. If X_mark is set to None,
            no mark is included.
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        projection = self.projection
        profile = self.profile
        from ._tools import show_projection
        return show_projection(projection, profile, shower_size, axes,
                               max_theta, X_mark)

    def show_profile(self):
        """
        Show the shower profile, both number of charged particles and energy
        deposit, as a function of slant depth.

        Returns
        -------
        (ax1, ax2) : AxesSubplot objects.
        """
        return self.profile.show()

    def show_light_production(self):
        """
        Show the production of both Cherenkov and fluorescence photons in the
        290 - 430 nm range as a function of slant depth.

        Returns
        -------
        (ax1, ax2) : AxesSubplot objects.
        """
        return self.shower.show_light_production()

    def show(self):
        """
        Show the signal evolution as a function of both time and beta angle
        (relative to the shower axis direction).

        The two contributions from Cherenkov and fluorescence light are shown.

        The time scale (us or ns) is adapted depending on the signal pulse
        duration.

        Returns
        -------
        (ax1, ax2) : AxesSubplot objects.
        """
        return _show(self)

    def Image(lat_profile=True, N_pix=None, int_time=0.01, NSB=40.):
        """
        Generate a time-varying shower image in a circular camera with square
        pixels of same solid angle. A Nishimura-Kamata-Greisen lateral profile
        is used to spread the signal contribution from each shower point to
        several pixels.

        Parameters
        ----------
        lat_profile : bool, default True
            Use a NKG lateral profile to spread the signal. If False, a linear
            shower is assumed.
        N_pix : int
            Number of camera pixels. If not given, the predefined value in
            the telescope that produces the signal.
        int_time : float
            Integration time in microseconds of a camera frame.
        NSB : float
            Night sky background in MHz/m$^2$/deg$^2$.

        Returns
        -------
        Image object
        """
        return sm.Image(self, N_pix, int_time, NSB)


# Auxiliary functions #########################################################
def _show(signal):
    track = signal.track
    projection = signal.projection

    points = signal.index
    distance = np.array(projection.distance.loc[points])
    if len(distance) == 0:
        print('The shower track is outside the telescope field of view.')
        return
    beta = np.array(projection.beta.loc[points])
    time = np.array(projection.time.loc[points])

    # Arrival time interval in microseconds (or nanoseconds) for each
    # discretization step.
    Delta_time = track.dl / 0.2998 * (1. - np.cos(np.radians(beta)))
    ns = True if time.max() < 0.1 else False  # Auto-scale
    if ns:
        time = time * 1000.
        Delta_time = Delta_time * 1000.

    # Number of photoelectrons due to each light component (and total)
    # per unit time
    cher_time = np.array(signal.Npe_cher / Delta_time)
    fluo_time = np.array(signal.Npe_fluo / Delta_time)
    total_time = cher_time + fluo_time

    # Number of photoelectrons due to each light componente (and total) per
    # unit of beta angle
    Delta_beta = np.degrees(track.dl / distance * np.sin(np.radians(beta)))
    cher_beta = np.array(signal.Npe_cher / Delta_beta)
    fluo_beta = np.array(signal.Npe_fluo / Delta_beta)
    total_beta = cher_beta + fluo_beta

    # Signal evolution as a function of time
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax1.plot(time, cher_time, 'r--', label='Cherenkov')
    ax1.plot(time, fluo_time, 'b--', label='Fluorescence')
    ax1.plot(time, total_time, 'k', label='Total')
    # Auto-scale
    if ns:
        ax1.axes.xaxis.set_label_text("Time (ns)")
        if signal.tel_eff:
            ax1.axes.yaxis.set_label_text("Photoelectrons / ns")
        else:
            ax1.axes.yaxis.set_label_text("Photons / ns")
    else:
        ax1.axes.xaxis.set_label_text("Time (us)")
        if signal.tel_eff:
            ax1.axes.yaxis.set_label_text("Photoelectrons / us")
        else:
            ax1.axes.yaxis.set_label_text("Photons / us")
    ax1.legend(loc=0)

    # Signal evolution as a function of beta
    ax2.plot(beta, cher_beta, 'r--', label='Cherenkov')
    ax2.plot(beta, fluo_beta, 'b--', label='Fluorescence')
    ax2.plot(beta, total_beta, 'k', label='Total')

    ax2.axes.xaxis.set_label_text("Beta (degrees)")
    if signal.tel_eff:
        ax2.axes.yaxis.set_label_text("Photoelectrons / degree")
    else:
        ax2.axes.yaxis.set_label_text("Photons / degree")
    ax2.legend(loc=0)
    plt.tight_layout()

    return (ax1, ax2)

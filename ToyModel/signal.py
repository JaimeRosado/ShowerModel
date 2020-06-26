# coding: utf-8

import math
import numpy as np
import pandas as pd
import ToyModel as tm
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
    projection : Projection object. If None, it will generated from telescope
        and shower.
    atm_trans : True if the atmospheric transmision is included to transport
        photons.
    tel_eff : True if the telescope efficiency is included to calculate the
        signal. If False, 100% efficiency is assumed for a given wavelenght
        interval.
    **kwargs {wvl_ini, wvl_fin, wvl_step}: Options to modify the wavelenght
        interval when tel_eff==False. If None, the wavelength interval defined
        in the telescope is used.

    Results
    -------
    Signal object.
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
        projection = tm.Projection(telescope, shower.track)
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
    Npe_cher : Number of photoelectrons per discretization step due to
        Cherenkov light.
    Npe_fluo : Number of photoelectrons per discretization step due to
        fluorescence light.
    Npe_total : Total number of photoelectrons per discretization step.

    Attributes
    ----------
    telescope : Telescope object.
    atm_trans : True if the atmospheric transmision is included.
    tel_eff : True if the telescope efficiency is included.
    wvl_ini : Initial wavelength in nm of the interval.
    wvl_fin : Final wavelength in nm of the interval.
    wvl_step : Step size in nm of the interval.
    Npe_cher_sum : Sum of photoelectrons due to Cherenkov light.
    Npe_fluo_sum : Sum of photoelectrons due to fluorescence light.
    Npe_total_sum : Sum of photoelectrons due to both light components.
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
    """

    def show_projection(self, shower_size=True, axes=True, max_theta=30.,
                        X_mark='X_max'):
        """
        Show the projection of the shower track viewed by the telescope in both
        horizontal and FoV coordinates systems.

        Parameters
        ----------
        shower_size : Bool indicating whether the radii of the shower track
            points are proportional to the shower size.
        axes : Bool indicating whether the axes of both frames of reference are
            visualized or not.
        max_theta : Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure, default to X_max. If X_mark is set to None,
            no mark is included.
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        projection = self.projection
        profile = self.profile
        from .tools import show_projection
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

def image(signal, NSB = 33.e6, NKG = True):
    """
    Generate a camera image in a projection with area proportional to solid
    angle, such that r = sqrt(1-cos(theta)).

    Parameters
    ----------
    NSB : Night sky background in MHz/m$^2$/sr.
    NKG : Bool indicating wether the shower lateral profile is included.
    """
    # Side of square pixel in solid angle projection
    Delta_pix = math.sqrt(telescope.sol_angle_pix / 2.)
    # Number of pixels across a radius within the camera FoV
    N_pix_r_exact = math.sqrt(telescope.N_pix / math.pi)
    N_pix_r = math.ceil(N_pix_r_exact)

    # Only points included in signal
    points = signal.index
    x = np.array(track.x.loc[points])
    y = np.array(track.y.loc[points])
    z = np.array(track.z.loc[points])
    s = np.array(profile.s.loc[points])
    # Apparent half lenght of a shower track step
    L = np.array(track.dl / 2.
                 * np.sin(np.radians(projection.beta.loc[points])))
    # Moliere radius
    R = np.array(atmosphere.r_M.loc[points])
    # FoV coordinates and photon arrival time
    distance = np.array(projection.distance.loc[points])
    cos_theta = np.array(np.cos(np.radians(projection.theta.loc[points])))
    # phi angle wrt right direction in radians
    phi = np.array(np.radians(projection.phi.loc[points]
                              - telescope.phi_right))  # (-2*pi, 2*pi)
    time = np.array(projection.time[points])
    # Total travel time of shower track within the camera
    Delta_time = time.max() - time.min()

    # Unit local coordinate vectors
    # Parallel to shower axis (upwards)
    ux = track.ux
    uy = track.uy
    uz = track.uz
    # Horizontal plane
    vx = track.vx
    vy = track.vy
    # vz = 0.  by definition
    # Perpendicular to both u and v
    wx = track.wx
    wy = track.wy
    wz = track.wz

    # Total number photoelectrons
    Npe = np.array(signal.Npe_total)
    # Night sky background per pixel
    NSB = NSB * 1.e-6 * telescope.area * telescope.sol_angle_pix * Delta_time

    # Approximate theta size of a pixel
    Delta_theta = 2. * Delta_pix / np.sqrt(1. + cos_theta)
    # Apparent angles of a cylinder coaxial to shower axis
    chi1 = np.arctan(L / distance)
    chi2 = np.arctan(R / distance)
    # Appararent size of the cylinder in number of pixels
    n1 = np.array(np.round(2. * chi1 / Delta_theta), int)  # Along shower axis
    n2 = np.array(np.round(2. * chi2 / Delta_theta), int)  # Perpendicular

    # Image size and initialization of pixel values
    image_clean = np.zeros((2*N_pix_r+1, 2*N_pix_r+1))
    image_noise = np.array(
        np.random.poisson(NSB, (2*N_pix_r+1, 2*N_pix_r+1)), float)
    for pix_y in range(2*N_pix_r+1):
        for pix_x in range(2*N_pix_r+1):
            if (pix_x-N_pix_r)**2 + (pix_y-N_pix_r)**2 > N_pix_r_exact**2:
                image_clean[pix_y, pix_x] = -float('inf')  # Blank pixels
                image_noise[pix_y, pix_x] = -float('inf')

    # Loop over shower points
    for point, (n1_p, n2_p, Npe_p) in enumerate(zip(n1, n2, Npe)):
        # No pixel spread
        if NKG and (n1_p < 2) and (n2_p < 2):
            phi_p = phi[point]
            # Radii and x, y indexes of pixels
            pix_r_p = math.sqrt(1. - cos_theta[point]) / Delta_pix
            pix_x_p = int(round(pix_r_p * np.cos(phi_p) + N_pix_r_exact))
            pix_y_p = int(round(pix_r_p * np.sin(phi_p) + N_pix_r_exact))
            image_clean[pix_y_p, pix_x_p] += Npe_p
            image_noise[pix_y_p, pix_x_p] += Npe_p
            continue

        # Lists of lenght = 1 to allow for loops
        x_p = [x[point]]
        y_p = [y[point]]
        z_p = [z[point]]

        # Contributions beyond one Moliere radius are not included
        # Num. photoelectrons are approximately corrected as a
        # function of shower age to account for these losses
        s_p = s[point]
        Npe_ps = Npe_p * (2. * s_p - 4.5) / (s_p - 4.5)

        # Apparent length of cylinder takes several pixels
        if n1_p > 1:
            # n1_p substeps along the shower axis
            Lmax = L[point]
            L_p = np.linspace(-Lmax, Lmax, n1_p)
            # Arrays of lenght = n1_p
            x_p = x_p + L_p * ux
            y_p = y_p + L_p * uy
            z_p = z_p + L_p * uz
            # Number of photoelectrons equally distributed in substeps
            Npe_ps = Npe_p / n1_p

            # A Moliere radius takes an only pixel
            if n2_p <2:
                # theta, phi are calculated for all substeps
                distance_p, alt_p, az_p, theta_p, phi_p = (
                    telescope.spherical(x_p, y_p, z_p))
                cos_theta_p = np.cos(np.radians(theta_p))
                phi_p = np.radians(phi_p - telescope.phi_right)
                # Radii and x, y indexes of pixels
                pix_r_p = np.sqrt(1. - cos_theta_p) / Delta_pix
                pix_x_p = np.array(np.round(pix_r_p * np.cos(phi_p)
                                            + N_pix_r_exact), int)
                pix_y_p = np.array(np.round(pix_r_p * np.sin(phi_p)
                                            + N_pix_r_exact), int)

                # Loop over substeps to spread signal over pixels
                for (pix_r_ps, pix_x_ps, pix_y_ps) in zip(pix_r_p, pix_x_p,
                                                          pix_y_p):
                    # Some substeps may lay outside the FoV
                    pix_r_ps = math.sqrt((pix_x_ps-N_pix_r)**2
                                         +(pix_y_ps-N_pix_r)**2)
                    if pix_r_ps <= pix_r_max:
                        image_clean[pix_y_ps, pix_x_ps] += Npe_ps
                        image_noise[pix_y_ps, pix_x_ps] += Npe_ps
                continue

        # A Moliere radius takes n2_p (>1) pixels
        Rmax = R[point]
        delta = 1./n2_p
        # n2_p radii are generated
        xx = np.arange(delta/2., 1., delta)  # In units of Moliere radius
        R_p = Rmax * xx
        # Each radius is weighted according to the NGG model
        weight_p = NKG(s_p, xx)
        Npe_ps = Npe_ps * weight_p / weight_p.sum()

        # Loop over substeps (may be only one)
        for (x_ps, y_ps, z_ps) in zip(x_p, y_p, z_p):
            # 5 * n2_p random polar angles for each substep
            alpha_ps = 2. * math.pi * np.random.rand(5, n2_p)
            # v, w projections of 5 * n2_p vectors
            Rv_ps = np.array((R_p * np.cos(alpha_ps)).flat)
            Rw_ps = np.array((R_p * np.sin(alpha_ps)).flat)
            # 5 * n2_p samples around shower axis
            x_ps = x_ps + Rv_ps * vx + Rw_ps * wx
            y_ps = y_ps + Rv_ps * vy + Rw_ps * wy
            z_ps = z_ps + Rw_ps * wz  # vz = 0 by definition
            # Fraction of photoelectrons for each polar angle
            fraction_ps = np.ones(5) / 5
            Npe_ps = np.array((Npe_ps * fraction_ps[:, np.newaxis]).flat)

            distance_ps, alt_ps, az_ps, theta_ps, phi_ps = (
                telescope.spherical(x_ps, y_ps, z_ps))
            cos_theta_ps = np.cos(np.radians(theta_ps))
            phi_ps = np.radians(phi_ps - telescope.phi_right)
            pix_r_ps = np.sqrt(1. - cos_theta_ps) / Delta_pix
            pix_x_ps = np.array(np.round(pix_r_ps * np.cos(phi_ps)
                                         + N_pix_r_exact), int)
            pix_y_ps = np.array(np.round(pix_r_ps * np.sin(phi_ps)
                                         + N_pix_r_exact), int)

            # Loop over samples around shower axis
            for (pix_r_pss, pix_x_pss, pix_y_pss,
                 Npe_pss) in zip(pix_r_ps, pix_x_ps, pix_y_ps, Npe_ps):
                pix_r_pss = math.sqrt((pix_x_pss-N_pix_r)**2
                                      +(pix_y_pss-N_pix_r)**2)
                if pix_r_pss <= pix_r_max:
                    image_clean[pix_y_pss, pix_x_pss] += Npe_pss
                    image_noise[pix_y_pss, pix_x_pss] += Npe_pss

    extent = (-N_pix_r-0.5, N_pix_r+0.5, -N_pix_r-0.5, N_pix_r+0.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax1.imshow(image_clean, extent=extent)  # cmap=viridis
    psm = ax2.imshow(image_noise, extent=extent)  # cmap=viridis
    cbar = fig.colorbar(psm)
    cbar.ax.set_ylabel('Photoelectrons');

def _NKG(s, x):
    """
    Non-normalized distribution of particles per unit radius according to the
    NKG model. 
    """
    s = s if s<2.25 else 2.24
    return x**(s-1.)*(1.+x)**(s-4.5)
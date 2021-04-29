# coding: utf-8

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


# Constructor #################################################################
def Image(signal, lat_profile=True, N_pix=None, int_time=None, NSB=40.):
    """
    Generate a time-varying shower image from a Signal object.

    A Nishimura-Kamata-Greisen lateral profile is used to spread the signal
    contribution from each shower point to several pixels. A circular camera with
    square pixels of same solid angle is assumed.

    Parameters
    ----------
    signal : Signal object.
    lat_profile : bool, default True
        Use a NKG lateral profile to spread the signal. If False, a linear
        shower is assumed.
    N_pix : int
        Number of camera pixels. If not given, the predefined value in the
        Telescope object is used.
    int_time : float
        Integration time in microseconds of a camera frame. If not
        given, the predefined value in the Telescope object is used.
    NSB : float
        Night sky background in MHz/m$^2$/deg$^2$.

    Returns
    -------
    image : Image object
    """
    image = _Image()
    image.NSB = NSB
    image.lat_profile = lat_profile
    image.signal = signal

    telescope = signal.telescope
    # Camera integration time
    if int_time is None:
        int_time = telescope.int_time
    image.int_time = int_time

    # Number of camera pixel and pixel solid angle
    if N_pix is None:
        N_pix = telescope.N_pix
        sol_angle_pix = telescope.sol_angle_pix
    else:
        sol_angle_pix = telescope.sol_angle / N_pix
    image.N_pix = N_pix
    image.sol_angle_pix = sol_angle_pix

    # Side of a square pixel in solid angle projection
    Delta_pix = math.sqrt(sol_angle_pix / 2.)
    # Number of pixels across a radius within the camera FoV
    N_pix_r_exact = math.sqrt(N_pix / math.pi)
    N_pix_r = math.ceil(N_pix_r_exact)
    image.N_pix_r = N_pix_r
    # Image size
    N = 2 * N_pix_r + 1

    # Night sky background per pixel and frame
    NSB_pix = (NSB * 90.**2 / math.pi**2 * telescope.area * sol_angle_pix
               * int_time)
    image.NSB_pix = NSB_pix

    # Only points included in signal
    points = signal.index
    if len(points) == 0:
        image.N_frames = 0
        image.frames  = np.zeros((0, N, N))
        return image

    # Parameters of contributing points
    track = signal.track
    projection = signal.projection
    atmosphere = signal.atmosphere
    profile = signal.profile
    x = np.array(track.x.loc[points])
    y = np.array(track.y.loc[points])
    z = np.array(track.z.loc[points])
    s = np.array(profile.s.loc[points])
    # Shower track half step and apparent size
    L_half = track.dl / 2.
    L = np.array(L_half * np.sin(np.radians(projection.beta.loc[points])))
    # Moliere radius
    R = np.array(atmosphere.r_M.loc[points])
    # FoV coordinates
    distance = np.array(projection.distance.loc[points])
    cos_theta = np.array(np.cos(np.radians(projection.theta.loc[points])))
    # phi angle wrt right direction in radians
    phi = np.array(np.radians(projection.phi.loc[points]
                              - telescope.phi_right))  # (-2*pi, 2*pi)
    # Total number of photoelectrons
    Npe = np.array(signal.Npe_total)

    # Arrival time of photons
    time = np.array(projection.time[points])
    # Frame index (>=0)
    t0 = time.min()
    f_index = np.array((time - t0) // int_time, int)
    # Total number of frames
    N_frames = f_index.max() + 1
    image.N_frames = N_frames
    # Initialization of pixel values at each frame
    frames = np.zeros((N_frames, N, N))
    # Camera FoV
    for pix_y in range(N):
        for pix_x in range(N):
            if (pix_x-N_pix_r)**2+(pix_y-N_pix_r)**2 > N_pix_r_exact**2:
                frames[:, pix_y, pix_x] = -float('inf')  # Blanck pixels

    if not lat_profile:
        # Radii and x, y indexes of pixels
        pix_r = np.sqrt(1. - cos_theta) / Delta_pix
        pix_x = np.array(np.round(pix_r * np.cos(phi) + N_pix_r_exact), int)
        pix_y = np.array(np.round(pix_r * np.sin(phi) + N_pix_r_exact), int)
        for (f_index_p, Npe_p, pix_x_p, pix_y_p) in zip(f_index, Npe, pix_x,
                                                        pix_y):
            frames[f_index_p, pix_y_p, pix_x_p] += Npe_p
        image.frames = frames
        return image

    # Approximate theta size of a pixel
    Delta_theta = 2. * Delta_pix / np.sqrt(1. + cos_theta)
    # Apparent angular size of a cylinder of half length L and radius R coaxial
    # to the shower axis at each point
    chi1 = 2. * np.arctan(L / distance)
    chi2 = 2. * np.arctan(R / distance)
    # Appararent size of the cylinder in number of pixels
    n1 = np.array(np.round(chi1 / Delta_theta), int)  # Along shower axis
    n2 = np.array(np.round(chi2 / Delta_theta), int)  # Perpendicular

    # Unit coordinate vectors
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

    # Loop over shower points
    for point, (f_index_p, n1_p, n2_p, Npe_p) in enumerate(zip(f_index, n1, n2,
                                                               Npe)):
        # If the apparent size of the cylinder is smaller than 1 pixel
        if (n1_p < 2) and (n2_p < 2):
            phi_p = phi[point]
            # Radii and x, y indexes of pixels
            pix_r_p = math.sqrt(1. - cos_theta[point]) / Delta_pix
            pix_x_p = int(round(pix_r_p * np.cos(phi_p) + N_pix_r_exact))
            pix_y_p = int(round(pix_r_p * np.sin(phi_p) + N_pix_r_exact))
            frames[f_index_p, pix_y_p, pix_x_p] += Npe_p
            continue

        # Lists of lenght = 1 to allow for loops
        x_p = [x[point]]
        y_p = [y[point]]
        z_p = [z[point]]

        # Contributions beyond one Moliere radius are not included
        # Number of photoelectrons are approximately corrected as a function of
        # shower age to account for these losses
        s_p = s[point]
        Npe_ps = Npe_p * (2. * s_p - 4.5) / (s_p - 4.5)

        # Apparent length of cylinder takes several pixels
        if n1_p > 1:
            # n1_p substeps along the shower axis
            L_p = np.linspace(-1., 1., n1_p) * L_half / 2.
            # Arrays of lenght = n1_p
            x_p = x_p + L_p * ux
            y_p = y_p + L_p * uy
            z_p = z_p + L_p * uz
            # Number of photoelectrons equally distributed in substeps
            Npe_ps = Npe_ps / n1_p

            # Apparent width of cylinder takes one only pixel
            if n2_p < 2:
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
                for (pix_x_ps, pix_y_ps) in zip(pix_x_p, pix_y_p):
                    # Some indexes may be outside the matrix bounds
                    if (pix_x_ps in range(N) and pix_y_ps in range(N)):
                        frames[f_index_p, pix_y_ps, pix_x_ps] += Npe_ps
                continue

        # Apparent width of cylinder takes several pixels
        delta = 1./n2_p
        # n2_p radii are generated
        xx = np.arange(delta/2., 1., delta)  # In units of Moliere radius
        R_p = R[point] * xx
        # Each radius is weighted according to the NKG model
        weight_p = _NKG(s_p, xx)
        Npe_ps = Npe_ps * weight_p / weight_p.sum()
        # 5 samples are generated for each radius
        Npe_ps = np.array((Npe_ps * np.full((5, n2_p), 1./5.)).flat)

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

            distance_ps, alt_ps, az_ps, theta_ps, phi_ps = (
                telescope.spherical(x_ps, y_ps, z_ps))
            cos_theta_ps = np.cos(np.radians(theta_ps))
            phi_ps = np.radians(phi_ps - telescope.phi_right)
            pix_r_ps = np.sqrt(1. - cos_theta_ps) / Delta_pix
            pix_x_ps = np.array(np.round(pix_r_ps * np.cos(phi_ps)
                                         + N_pix_r_exact), int)
            pix_y_ps = np.array(np.round(pix_r_ps * np.sin(phi_ps)
                                         + N_pix_r_exact), int)

            # Loop over samples points around shower axis
            for (pix_x_pss, pix_y_pss, Npe_pss) in zip(pix_x_ps, pix_y_ps,
                                                       Npe_ps):
                # Some indexes may be outside the matrix bounds
                if (pix_x_pss in range(N) and pix_y_pss in range(N)):
                    frames[f_index_p, pix_y_pss, pix_x_pss] += Npe_pss

    image.frames = frames
    return image


# Class #######################################################################
class _Image:
    """
    Object containing a time-varying shower image in a circular camera with
    square pixels of same solid angle. A Nishimura-Kamata-Greisen lateral
    profile is used to spread the signal contribution from each shower point to
    several pixels.

    Attributes
    ----------
    signal : Signal object.
    lat_profile : bool
        Bool indicating wether a NKG lateral profile is used to
        spread the signal. If False, a linear shower is assumed.
    N_pix : int
        Number of camera pixels.
    N_pix_r : int
        Number of pixels across a camera radius.
    sol_angle_pix : float
        Solid angle in stereoradians of a single pixel.
    int_time : float
        Integration time in microseconds of a camera frame.
    N_frames : int
        Number of frames.
    frames : array
        Array of size (N_frames, 2*N_pix_r+1, 2*N_pix_r+1) containing the
        pixel values at each frame. Array elements not corresponding to any
        camera pixel are set to -inf.
    NSB : float
        Night sky background in MHz/m$^2$/deg$^2$.
    NSB_pix : float
        Mean number of background photoelectrons per pixel and frame.

    Methods
    -------
    show : Show a camera frame or the sum of all them including random
        background.
    animate : Show an animation of camera frames.
    """
    pass

    # Methods #################################################################
    def show(self, frame=None, NSB=None, ax=None):
        """
        Show a camera frame or the sum of all them including random background.

        Parameters
        ----------
        frame : int
            Frame number. If not given, the sum of frames is shown.
        NSB : float
            Night sky background in MHz/m$^2$/deg$^2$. If not given, the one
            defined in Image is used.
        ax : AxesSubplot object
            Axes instance where the plot is generated. In not given, a new
            AxesSubplot object is created.

        Returns
        -------
        axes : AxesSubplot object.
        """
        N_pix = self.N_pix
        N_pix_r = self.N_pix_r
        N_frames = self.N_frames
        frames = self.frames
        # Image size
        N = 2 * N_pix_r +1

        # If an Axes instance is not given, a new AxesSubplot is created
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # If no signal, an empty plot is generated
        if N_frames == 0:
            ax.text(0.4, 0.5, 'No signal')
            return ax

        # The NSB defined in image is used
        if NSB is None:
            NSB_pix = self.NSB_pix
        # The input NSB is used
        else:
            telescope = self.signal.telescope
            int_time = self.int_time
            sol_angle_pix = self.sol_angle_pix
            NSB_pix = (NSB * 90.**2 / math.pi**2 * telescope.area *
                       sol_angle_pix * int_time)

        # Sum of frames
        if frame is None:
            image = frames.sum(0)
            # Integrated noise
            noise = np.array(np.random.poisson(N_frames*NSB_pix, (N, N)),
                             float)
            image += noise
        # A given frame is shown
        elif frame < N_frames:
            image = frames[frame]
            # Noise of a single frame
            noise = np.array(np.random.poisson(NSB_pix, (N, N)), float)
            image += noise
        else:
            raise ValueError(
                'The frame number must be lower the number of frames')

        extent = (-N_pix_r-0.5, N_pix_r+0.5, -N_pix_r-0.5, N_pix_r+0.5)
        psm = ax.imshow(image, extent=extent)  # cmap=viridis
        plt.colorbar(psm, ax=ax, label='Photoelectrons')
        return ax

    def animate(self, NSB=None):
        """
        Show an interactive animation of camera frames.

        Parameters
        ----------
        NSB : float
            Night sky background in MHz/m$^2$/deg$^2$. If not given, the one
            defined in Image is used.

        Returns
        -------
        ani : HTML object.
        """
        N_pix = self.N_pix
        N_pix_r = self.N_pix_r
        N_frames = self.N_frames
        frames = self.frames
        # Image size
        N = 2 * N_pix_r +1

        # If no signal, an empty plot is generated
        if N_frames == 0:
            fig = plt.figure(figsize=(5, 5))
            plt.text(0.4, 0.5, 'No signal')
            return fig

        # The NSB defined in image is used
        if NSB is None:
            NSB_pix = self.NSB_pix
        # The input NSB is used
        else:
            telescope = self.signal.telescope
            int_time = self.int_time
            sol_angle_pix = self.sol_angle_pix
            NSB_pix = (NSB * 90.**2 / math.pi**2 * telescope.area *
                       sol_angle_pix * int_time)

        fig = plt.figure()
        extent = (-N_pix_r-0.5, N_pix_r+0.5, -N_pix_r-0.5, N_pix_r+0.5)
        # Same color scale for all frames
        vmax = frames.max()
        vmin = 0.
        # List of AxesSubplot objects to pass to ArtistAnimation function
        ims = []
        for i, f in enumerate(frames):
            # Noise of a single frame
            noise = np.array(
                np.random.poisson(NSB_pix, (N, N)), float)
            f += noise

            im = plt.imshow(f, extent=extent, vmin=vmin, vmax=vmax)
            ims.append([im])

        # One only colorbar
        cbar = fig.colorbar(im)
        cbar.ax.set_ylabel('Photoelectrons');

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                        repeat_delay=500)
        # Interactive HTML object
        ani = HTML(ani.to_jshtml())
        return ani


# Auxiliary functions #########################################################
def _NKG(s, x):
    """
    Non-normalized distribution of particles per unit radius according to the
    NKG model.
    """
    s = s if s < 2.25 else 2.24
    return x**(s-1.)*(1.+x)**(s-4.5)

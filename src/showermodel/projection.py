# coding: utf-8

import numpy as np
import pandas as pd
import showermodel.constants as ct

# Class #######################################################################
class Projection(pd.DataFrame):
    """
    DataFrame containing the projection of a shower track.

    The track is viewed from a telescope position in both horizontal
    coordinates system and FoV coordinates system as well as the fraction of
    the track within the telescope field of view.

    Parameters
    ----------
    telescope : Telescope, mandatory
        Telescope object to be used.
    track : Track or Shower, mandatory
        Track object to be used.

    Attributes
    ----------
    distance : float
        Column 0, shower-to-telescope distance in km.
    alt : float
        Column 1, altitude in degrees (from horizon).
    az : float
        Column 2, azimuth in degrees (from north, clockwise).
    theta : float
        Column 3, offset angle in degrees relative to the telescope pointing
        direction.
    phi : float
        Column 4, position angle in degrees from north in FoV projection.
    beta : float
        Column 5, angle in degrees relative to the apparent source position.
    time : float
        Column 6, arrival time in microseconds of photons emitted at each point
        of the shower, where time=0 for photons produced at the top of the
        atmosphere.
    FoV : bool
        Column 7, True if the shower point is within the telescope field of
        view, False otherwise.
    atmosphere : Atmosphere
        Atmosphere object that is used.
    track : Track
        Track object that is used.
    telescope : Telescope
        Telescope object that is used.
    distance_top, alt_top, ..., beta_top : float or None
        Coordinates of the shower point at the top of the atmosphere (if any).
    distance_0, alt_0, ..., beta_0 : float or None
        Coordinates of the shower impact point at ground (if any).
    distance_i, alt_i, ..., beta_i : float or None
        Coordinates of the first interaction point of the shower.
    distance_min : float
        Minimum distance in km to (infinite) line going to the
        shower axis.
    alt_inf : float
        Altitude in degrees of the apparent source position.
    az_inf : float
        Azimuth in degrees of the apparent source position.
    theta_inf : float
        Offset angle in degrees of the apparent source position.
    phi_inf : float
        Position angle in degrees of the apparent source position.

    Methods
    -------
    show()
        Show the projection of the shower track viewed by the telescope.
    hor_to_FoV()
        Convert cartesian coordinates from horizontal system to FoV system.
    FoV_to_hor()
        Convert cartesian coordinates from FoV system to horizontal system.
    theta_phi_to_alt_az()
        Convert FoV coordinates theta/phi to horizontal coordinates alt/az.
    altaz_to_thetaphi()
        Convert horizontal coordinates alt/az to FoV coordinates theta/phi.
    spherical()
        Calculate the spherical coordinates in both horizontal and FoV systems.
    """
    def __init__(self, telescope, track):
        super().__init__(columns=['distance', 'alt', 'az', 'theta', 'phi',
                                  'beta', 'time', 'FoV'])
        _projection(self, telescope, track)

    def show(self, axes=True, max_theta=30., X_mark=None):
        """
        Show the projection of the shower track viewed by the telescope in both
        horizontal and FoV coordinates systems.

        Parameters
        ----------
        axes : bool, default True
            Show the axes of both frames of reference.
        max_theta : float, default 30
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float, default None
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure. If None, no mark is included.

        Returns
        -------
        ax1, ax2 : PolarAxesSubplot
        """
        from ._tools import show_projection
        return show_projection(self, None, False, axes, max_theta, X_mark)

    def altaz_to_thetaphi(self, alt, az):
        """
        Convert polar horizontal coordinates alt, az to FoV coordinates
        theta, phi.

        Parameters
        ----------
        alt, az : float or array_like

        Returns
        -------
        theta, phi : float or array-like

        See also
        --------
        Projection.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Projection.theta_phi_to_alt_az : Convert FoV coordinates theta, phi to
            horizontal coordinates alt, az.
        """
        return self.telescope.altaz_to_thetaphi(alt, az)

    def hor_to_FoV(self, x_hor, y_hor, z_hor):
        """
        Convert cartesian coordinates from horizontal system to FoV system.

        In the FoV coordinates system, x_FoV grows in the right-hand direction,
        y_FoV grows downward and z_FoV grows toward the pointing direction from
        the telescope point of view.

        Parameters
        ----------
        x_hor, y_hor, z_hor : float or array-like

        Returns
        -------
        x_FoV, y_FoV, z_FoV : floar or array_like

        See also
        --------
        Projection.FoV_to_hor : Convert cartesian coordinates from FoV system
            to horizontal system.
        Projection.altaz_to_thetaphi : Convert horizontal coordinates alt, az
            to FoV coordinates theta, phi.
        """
        return self.telescope.hor_to_FoV(x_hor, y_hor, z_hor)

    def theta_phi_to_alt_az(self, theta, phi):
        """
        Convert FoV coordinates theta, phi to horizontal coordinates alt, az.

        Parameters
        ----------
        theta, phi : float or array_like

        Returns
        -------
        alt, az : float or array_like

        See also
        --------
        Projection.FoV_to_hor : Convert cartesian coordinates from FoV system
            to horizontal system.
        Projection.altaz_to_thetaphi : Convert horizontal coordinates alt, az
            to FoV coordinates theta, phi.
        """
        return self.telescope.theta_phi_to_alt_az(theta, phi)

    def FoV_to_hor(self, x_FoV, y_FoV, z_FoV):
        """
        Convert cartesian coordinates from FoV system to horizontal system.

        In the FoV coordinates system, x_FoV grows in the right-hand direction,
        y_FoV grows downward and z_FoV grows toward the pointing direction from
        the telescope point of view.

        Parameters
        ----------
        x_FoV, y_FoV, z_FoV : float or array_like

        Returns
        -------
        x_hor, y_hor, z_hor : float or array_like.

        See also
        --------
        Projection.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Projection.theta_phi_to_alt_az : Convert FoV coordinates theta, phi to
            horizontal coordinates alt, az.
        """
        return self.telescope.FoV_to_hor(x_FoV, y_FoV, z_FoV)

    def spherical(self, x, y, z):
        """
        Calculate the spherical coordinates in both horizontal and FoV systems
        from the 'absolute' x, y, z coordinates.

        Parameters
        ----------
        x, y, z : float or array_like

        Returns
        -------
        distance, alt, az, theta, phi : float or array_like
        """
        return self.telescope.spherical(x, y, z)


# Constructor #################################################################
def _projection(projection, telescope, track):
    """
    Obtain the projection of a shower track viewed from the telescope position
    in both horizontal coordinates system (alt/az) and FoV coordinates system
    (theta/phi) and determine the fraction of the track within the telescope
    field of view.

    Parameters
    ----------
    projection : Projection
    telescope : Telescope
    track : Track or Shower
    """
    from .telescope import Telescope
    from .track import Track
    from .shower import Shower
    if isinstance(telescope, Telescope):
        pass
    # In case the input objects are not ordered correctly.
    elif isinstance(telescope, Track):
        telescope, track = (track, telescope)
    elif isinstance(telescope, Shower):
        telescope, shower = (track, telescope)
        track = shower.track
    else:
        raise ValueError('The input telescope is not valid')
    if isinstance(track, Track):
        pass
    elif isinstance(track, Shower):
        shower = track
        track = shower.track
    else:
        raise ValueError('The input track is not valid')

    # projection = Projection(columns=['distance', 'alt', 'az', 'theta', 'phi',
    #                                   'beta', 'time', 'FoV'])
    projection.atmosphere = track.atmosphere
    projection.track = track
    projection.telescope = telescope

    # Apparent position of the cosmic-ray source
    projection.alt_inf = track.alt
    projection.az_inf = track.az
    theta_inf, phi_inf = telescope.altaz_to_thetaphi(track.alt, track.az)
    projection.theta_inf = theta_inf
    projection.phi_inf = phi_inf

    # Shower spherical coordinates in both zenith and camera projections
    distance, alt, az, theta, phi = telescope.spherical(track.x, track.y,
                                                        track.z)
    projection.distance = distance
    projection.alt = alt
    projection.az = az
    projection.theta = theta
    projection.phi = phi

    # Coordinates of first interaction point of the shower relative to
    # the telescope position
    distance_i, alt_i, az_i, theta_i, phi_i = telescope.spherical(track.xi,
                                              track.yi, track.zi)
    projection.distance_i = distance_i
    projection.alt_i = alt_i
    projection.az_i = az_i
    projection.theta_i = theta_i
    projection.phi_i = phi_i
    # Angle formed by the shower axis (backwards) and the vector going
    # from the telescope position to the first interaction point
    xi, yi, zi = telescope.abs_to_rel(track.xi, track.yi, track.zi)
    proj_u_i = xi * track.ux + yi * track.uy + zi * track.uz
    beta_i = telescope.zr_to_theta(proj_u_i, distance_i)

    # Coordinates of the shower point at the top of the atmosphere relative to
    # the telescope position
    if track.z_top is None:
        distance_top = None
        beta_top = None
        projection.alt_top = None
        projection.az_top = None
        projection.theta_top = None
        projection.phi_top = None
    elif track.z_top==track.zi:
        distance_top = distance_i
        proj_u_top = proj_u_i
        beta_top = beta_i
        projection.alt_top = alt_i
        projection.az_top = az_i
        projection.theta_top = theta_i
        projection.phi_top = phi_i
    else:
        distance_top, alt_top, az_top, theta_top, phi_top = (
            telescope.spherical(track.x_top, track.y_top, track.z_top))
        projection.alt_top = alt_top
        projection.az_top = az_top
        projection.theta_top = theta_top
        projection.phi_top = phi_top
        # Angle formed by the shower axis (backwards) and the vector going
        # from the telescope position to the first interaction point
        x_top, y_top, z_top = telescope.abs_to_rel(track.x_top, track.y_top,
                                                   track.z_top)
        proj_u_top = x_top * track.ux + y_top * track.uy + z_top * track.uz
        beta_top = telescope.zr_to_theta(proj_u_top, distance_top)
    projection.distance_top = distance_top

    # Coordinates of the shower impact point at ground level relative to
    # the telescope position and minimum shower-to-telescope distance
    if track.z0 is None:
        distance_0 = None
        beta_0 = None
        projection.alt_0 = None
        projection.az_0 = None
        projection.theta_0 = None
        projection.phi_0 = None
        # Minimum shower-to-telescope distance
        distance_min = distance_i * np.sin(np.radians(beta_i))
    elif track.z0==track.zi:
        distance_0 = distance_i
        proj_u_0 = proj_u_i
        beta_0 = beta_i
        projection.alt_0 = alt_i
        projection.az_0 = az_i
        projection.theta_0 = theta_i
        projection.phi_0 = phi_i
        # Minimum shower-to-telescope distance
        distance_min = distance_i * np.sin(np.radians(beta_i))
    else:
        distance_0, alt_0, az_0, theta_0, phi_0 = telescope.spherical(track.x0,
                                                  track.y0, track.z0)
        projection.alt_0 = alt_0
        projection.az_0 = az_0
        projection.theta_0 = theta_0
        projection.phi_0 = phi_0
        # Angle formed by the shower axis (backwards) and the vector going
        # from the telescope position to the shower impact point at ground
        x0, y0, z0 = telescope.abs_to_rel(track.x0, track.y0, track.z0)
        proj_u_0 = x0 * track.ux + y0 * track.uy + z0 * track.uz
        beta_0 = telescope.zr_to_theta(proj_u_0, distance_0)
        if distance_0<distance_i:
            # Minimum shower-to-telescope distance
            distance_min = distance_0 * np.sin(np.radians(beta_0))
        else:
            distance_min = distance_i * np.sin(np.radians(beta_i))
    projection.distance_0 = distance_0

    # Half radius of the telescope mirror in km
    half_R = np.sqrt(telescope.area / ct.pi) / 2000.
    # If the telescope is too close to the shower axis
    x, y, z = telescope.abs_to_rel(track.x, track.y, track.z)
    proj_u = x * track.ux + y * track.uy + z * track.uz
    if distance_min < half_R:
        # Force minimum beta due the finite dimensions of the telescope mirror
        beta = telescope.xy_to_phi(proj_u, half_R)
        beta_i = telescope.xy_to_phi(proj_u_i, half_R)
        if track.z0 is not None:
            beta_0 = telescope.xy_to_phi(proj_u_0, half_R)
        if track.z_top is not None:     
            beta_top = telescope.xy_to_phi(proj_u_top, half_R)
    else:
        beta = telescope.zr_to_theta(proj_u, distance)

    projection.beta = beta
    projection.beta_i = beta_i
    projection.beta_0 = beta_0
    projection.beta_top = beta_top

    # Travel time of photons reaching the telescope, with time=0 for photons
    # emitted from the first interaction point. Equivalent to
    # projection.time = track.t - (distance_i - distance) / ct.c_km_us
    # except for distance_min<half_R
    # c_km_us: speed of light in km/us
    projection.time = (track.t - distance_i / ct.c_km_us
                       * (1. - np.sin(np.radians(projection.beta_i))
                       / np.sin(np.radians(beta))))

    # FoV = True for shower points within the telescope field of view
    projection.FoV = ((projection.theta <= telescope.apert/2.)
                      & (projection.distance > 0.))

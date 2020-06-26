# coding: utf-8

import numpy as np
import pandas as pd


# Constructor #################################################################
def Projection(telescope, track):
    """
    Obtain the projection of a shower track viewed from the telescope position
    in both horizontal coordiantes system (alt/az) and FoV coordinates system
    (theta/phi) and determine the fraction of the track within the telescope
    field of view.

    Parameters
    ----------
    telescope : Telescope object.
    track : Track object or Shower object.

    Returns
    -------
    Projection object.
    """
    from .telescope import _Telescope
    from .track import _Track
    from .shower import _Shower
    if isinstance(telescope, _Telescope):
        pass
    # In case the input objects are not ordered correctly.
    elif isinstance(telescope, _Track):
        telescope, track = (track, telescope)
    elif isinstance(telescope, _Shower):
        telescope, shower = (track, telescope)
        track = shower.track
    else:
        raise ValueError('The input telescope is not valid')
    if isinstance(track, _Track):
        pass
    elif isinstance(track, _Shower):
        shower = track
        track = shower.track
    else:
        raise ValueError('The input track is not valid')

    projection = _Projection(columns=['distance', 'alt', 'az', 'theta', 'phi',
                                      'beta', 'time', 'FoV'])
    projection.atmosphere = track.atmosphere
    projection.track = track
    projection.telescope = telescope

    # Shower spherical coordinates in both zenith and camera projections
    distance, alt, az, theta, phi = telescope.spherical(track.x, track.y,
                                                        track.z)
    projection.distance = distance
    projection.alt = alt
    projection.az = az
    projection.theta = theta
    projection.phi = phi

    # Coordinates of the shower impact point at ground level relative to
    # the telescope position
    distance_0, alt_0, az_0, theta_0, phi_0 = telescope.spherical(track.x0,
                                                                  track.y0, 0.)
    projection.distance_0 = distance_0
    projection.alt_0 = alt_0
    projection.az_0 = az_0
    projection.theta_0 = theta_0
    projection.phi_0 = phi_0

    # Coordinates of the shower point at the top of the atmosphere relative to
    # the telescope position
    distance_top = telescope.distance(track.x_top, track.y_top, track.z_top)
    projection.distance_top = distance_top

    # Apparent position of the cosmic-ray source
    projection.alt_inf = track.alt
    projection.az_inf = track.az
    theta_inf, phi_inf = telescope.altaz_to_thetaphi(track.alt, track.az)
    projection.theta_inf = theta_inf
    projection.phi_inf = phi_inf

    # Angle formed by the shower axis (upwards) and the vector going from the
    # telescope position to the shower impact point at ground
    x0, y0, z0 = telescope.abs_to_rel(track.x0, track.y0, 0.)
    beta_0 = telescope.zr_to_theta(x0 * track.ux + y0 * track.uy
                                   + z0 * track.uz, distance_0)
    # Minimum shower-to-telescope distance
    distance_min = distance_0 * np.sin(np.radians(beta_0))
    projection.distance_min = distance_min
    # Half radius of the telescope mirror in km
    half_R = np.sqrt(telescope.area / np.pi) / 2000.
    # If the telescope is too close to the shower axis
    if distance_min < half_R:
        # Force minimum beta due the finite dimensions of the telescope mirror
        beta = telescope.xy_to_phi(distance, half_R)
        projection.beta = beta
        projection.beta_0 = telescope.xy_to_phi(distance_0, half_R)
        projection.beta_top = telescope.xy_to_phi(distance_top, half_R)
    else:
        x, y, z = telescope.abs_to_rel(track.x, track.y, track.z)
        beta = telescope.zr_to_theta(x * track.ux + y * track.uy
                                     + z * track.uz, distance)
        projection.beta = beta
        projection.beta_0 = beta_0
        x_top, y_top, z_top = telescope.abs_to_rel(track.x_top, track.y_top,
                                                   track.z_top)
        projection.beta_top = (
            telescope.zr_to_theta(x_top * track.ux + y_top * track.uy
                                  + z_top * track.uz, distance_top))

    # Travel time of photons reaching the telescope, with time=0 for photons
    # emitted at the top of the atmosphere. Equivalent to
    # projection.time = track.t - (distance_top - distance) / 0.2998
    # except for distance_min<half_R
    projection.time = (track.t - distance_top / 0.2998
                       * (1. - np.sin(np.radians(projection.beta_top))
                          / np.sin(np.radians(beta))))

    # FoV = True for shower points within the telescope field of view
    projection.FoV = ((projection.theta <= telescope.apert/2.)
                      & (projection.distance > 0.))

    return projection


# Class #######################################################################
class _Projection(pd.DataFrame):
    """
    DataFrame containing the projection of a shower track viewed from a
    telescope poisition in both horizontal coordinates system and FoV
    coordinates system as well as the fraction of the track within the
    telescope field of view.

    Columns
    -------
    distance : Shower-to-telescope distance in km.
    alt : Altitude in degrees (from horizon).
    az : Azimuth in degrees (from north, clockwise).
    theta : Offset angle in degrees relative to the telescope pointing
        direction.
    phi : Position angle in degrees from north in FoV projection.
    beta : Angle in degrees relative to the apparent source position.
    time : Arrival time in microseconds of photons emitted at each point of
        the shower, where time=0 for photons produced at the top of the
        atmosphere.
    FoV : True if the shower point is within the telescope field of view,
        False otherwise.

    Attributes
    ----------
    atmosphere : Atmosphere object.
    track : Track object.
    telescope : Telescope object.
    distance_top : Distance in km to shower point at the top of the atmosphere.
    beta_top : Beta angle in degrees of the shower point at the top of the
        atmosphere.
    distance_0 : Distance in km to the shower impact point at ground.
    beta_0 : Beta angle in degrees of the shower impact point at ground.
    distance_min : Minimum distance in km to (infinite) line going to the
        shower axis.
    alt_inf : Altitude in degrees of the apparent source position.
    az_inf : Azimuth in degrees of the apparent source position.
    theta_inf : Offset angle in degrees of the apparent source position.
    phi_inf : Position angle in degrees of the apparent source position.

    Methods
    -------
    show : Show the projection of the shower track viewed by the telescope in
        both horizontal and FoV coordinates systems.
    hor_to_FoV : Convert cartesian coordinates from horizontal system to
        FoV system.
    FoV_to_hor : Convert cartesian coordinates from FoV system to horizontal
        system.
    thetaphi_to_altaz : Convert FoV coordinates theta/phi to horizontal
        coordinates alt/az.
    altaz_to_thetaphi : Convert horizontal coordinates alt/az to FoV
        coordinates theta/phi.
    spherical : Calculate the spherical coordinates in both horizontal and
        FoV systems.
    """
    def show(self, axes=True, max_theta=30., X_mark=None):
        """
        Show the projection of the shower track viewed by the telescope in both
        horizontal and FoV coordinates systems.

        Parameters
        ----------
        axes : Bool indicating whether the axes of both frames of reference are
            visualized or not.
        max_theta : Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure. If None, no mark is included.

        Returns
        -------
        ax1, ax2 : PolarAxesSubplot objects.
        """
        from .tools import show_projection
        return show_projection(self, None, False, axes, max_theta, X_mark)

    def altaz_to_thetaphi(self, alt, az):
        """
        Convert polar horizontal coordinates alt, az to FoV coordinates
        theta, phi.

        Parameters
        ----------
        alt : Scalar or array-like.
        az : Scalar or array-like.

        Returns
        -------
        theta, phi : Scalars or array-like objects.

        See also
        --------
        Projection.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Projection.thetaphi_to_altaz : Convert FoV coordinates theta, phi to
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
        x_hor : Scalar or array-like.
        y_hor : Scalar or array-like.
        z_hor : Scalar or array-like.

        Returns
        -------
        x_FoV, y_FoV, z_FoV : Scalars or array-like objects.

        See also
        --------
        Projection.FoV_to_hor : Convert cartesian coordinates from FoV system
            to horizontal system.
        Projection.altaz_to_thetaphi : Convert horizontal coordinates alt, az
            to FoV coordinates theta, phi.
        """
        return self.telescope.hor_to_FoV(x_hor, y_hor, z_hor)

    def thetaphi_to_altaz(self, theta, phi):
        """
        Convert FoV coordinates theta, phi to horizontal coordinates alt, az.

        Parameters
        ----------
        theta : Scalar or array-like.
        phi : Scalar or array-like.

        Returns
        -------
        alt, az : Scalars or array-like objects.

        See also
        --------
        Projection.FoV_to_hor : Convert cartesian coordinates from FoV system
            to horizontal system.
        Projection.altaz_to_thetaphi : Convert horizontal coordinates alt, az
            to FoV coordinates theta, phi.
        """
        return self.telescope.thetaphi_to_altaz(theta, phi)

    def FoV_to_hor(self, x_FoV, y_FoV, z_FoV):
        """
        Convert cartesian coordinates from FoV system to horizontal system.

        In the FoV coordinates system, x_FoV grows in the right-hand direction,
        y_FoV grows downward and z_FoV grows toward the pointing direction from
        the telescope point of view.

        Parameters
        ----------
        x_FoV : Scalar or array-like.
        y_FoV : Scalar or array-like.
        z_FoV : Scalar or array-like.

        Returns
        -------
        x_hor, y_hor, z_hor : Scalars or array-like objects.

        See also
        --------
        Projection.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Projection.thetaphi_to_altaz : Convert FoV coordinates theta, phi to
            horizontal coordinates alt, az.
        """
        return self.telescope.FoV_to_hor(x_FoV, y_FoV, z_FoV)

    def spherical(self, x, y, z):
        """
        Calculate the spherical coordinates in both horizontal and FoV systems
        the 'absolute' x, y, z coordinates.

        Parameters
        ----------
        x, y, z : Scalars or array-like objects.

        Returns
        -------
        distance, alt, az, theta, phi : Scalars or array-like objects.
        """
        return self.telescope.spherical(x, y, z)

# coding: utf-8

import math
import numpy as np
import pandas as pd
import showermodel as sm
import warnings
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)


# Default values for track
_theta = 0.  # deg
_az = 0.     # deg
_x0 = 0.     # km
_y0 = 0.     # km


# Constructor #################################################################
def Track(theta=_theta, alt=None, az=_az, x0=_x0, y0=_y0, atmosphere=None,
          **kwargs):
    """
    Make a shower track discretization.

    Parameters
    ----------
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apperent position of the source. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0 : float
        East coordinate in km of shower impact point at ground.
    y0 : float
        West coordinate in km of shower impact point at ground.
    atmosphere : Atmosphere object
        If None, a new Atmosphere object is generated.
    **kwargs {h0, h_top, N_steps, model}
        Options to construct the new Atmosphere object when atm==None.
        If None, the default Atmosphere object is used.

    Returns
    -------
    track : Track object.

    See also
    --------
    _Track : Track class.
    Shower : Contructor of Shower object.
    """
    from .atmosphere import _Atmosphere
    if isinstance(atmosphere, _Atmosphere):
        pass
    elif atmosphere is None:
        atmosphere = sm.Atmosphere(**kwargs)
    else:
        raise ValueError('The input atmosphereis not valid.')

    # The columns of the output DataFrame includes, at each discretization step
    # of are: coordinates (x,y,z) in km and the travel time t in us
    track = _Track(columns=['x', 'y', 'z', 't'])
    track.atmosphere = atmosphere

    # The input parameters along with some geometric parameters are also
    # included as atributes of the DataFrame. The angles are stored in degrees
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    track.theta = theta
    track.alt = alt
    theta = math.radians(theta)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    track.az = az
    az = math.radians(az)
    cos_az = math.cos(az)
    sin_az = math.sin(az)

    # Coordinates of the unit vector pointing at the arrival shower direction
    track.ux = sin_theta * sin_az
    track.uy = sin_theta * cos_az
    track.uz = cos_theta

    # Coordinates of a unit vector perpendicular to u and parallel to
    # horizontal xy plane
    track.vx = cos_az
    track.vy = -sin_az
    track.vz = 0.

    # Coordinates of the unit vector perpendicular to both u and v.
    track.wx = cos_theta * sin_az
    track.wy = cos_theta * cos_az
    track.wz = -sin_theta

    # Core position at ground level (z=0) and on the top of the atmosphere
    # (z=zmax)
    track.x0 = x0
    track.y0 = y0
    track.z_top = atmosphere.h_top - atmosphere.h0
    track.x_top = x0 + track.z_top * track.ux / track.uz
    track.y_top = y0 + track.z_top * track.uy / track.uz

    # Total travel time in us, where t=0 corresponds to the moment when the
    # shower enters the atmosphere
    track.t0 = track.z_top / track.uz / 0.2998

    # Distance in km travelled trhough one atmospheric slice
    track.dl = atmosphere.h_step / track.uz

    # Coordinates along the shower track
    track.z = atmosphere.h - atmosphere.h0
    track.x = x0 + track.z * track.ux / track.uz
    track.y = y0 + track.z * track.uy / track.uz
    track.t = (track.z_top - track.z) / track.uz / 0.2998  # Travel time

    return track


# Class #######################################################################
class _Track(pd.DataFrame):
    """
    DataFrame containing a linear-shower track discretization.

    Use Track to construct a Track object.

    Columns
    -------
    x : float
        East coordinate in km.
    y : float
        North coordinate in km.
    z : float
        Height in km from ground level.
    t : float
        Travel time in microseconds. t=0 at the top of the atmosphere. The
        shower is assumed to propates with the speed of light.

    Attributes
    ----------
    atmosphere : Atmosphere object.
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    ux : float
        East coordinate of a unit vector parallel to  the shower axis
        (upwards).
    uy : float
        North coordinate of a unit vector parallel to the shower axis
        (upwards).
    uz : float
        Vertical coordinate of a unit vector parallel to the shower axis
        (upwards).
    vx : float
        East coordinate of a unit vector perpendicular to shower axis and
        parallel to horizontal plane.
    vy : float
        North coordinate of a unit vector perpendicular to shower axis and
        parallel to horizontal plane.
    vz : =0. always
        Vertical coordinate of vector v.
    wx : float
        East coordinate of a unit vector perpendicular to both u and v.
    wy : float
        North coordinate of a unit vector perpendicular to both u and v.
    wz : float
        Vertical coordinate of a unit vector perpendicular to both u and v.
    x0 : float
        East coordinate in km of shower impact point at ground.
    y0 : float
        North coordinate in km of shower impact point at ground.
    t0 : float
        Travel time in microseconds at ground level.
    x_top : float
        East coordinate in km of shower at the top of the atmosphere.
    y_top : float
        North coordinate in km of shower at the top of the atmosphere.
    z_top : float
        Height in km of the top of the atmosphere from ground level.
    dl : float
        Size in km of discretization step along the shower axis.

    Methods
    -------
    h_to_xyz : Get the spatial coordinates from height above sea level.
    z_to_t : Get travel time mass density from height.
    X_to_xyz : Get the spatial coordinates from slanth depth.
    Projection : Make a Projection object containing the coordinates of a
        shower track relative to a telescope position.
    show_projection : Make a Projection object and show it.

    See also
    --------
    Track : Constructor of Track object.
    Shower : Constructor of Shower object.
    """

    def h_to_xyz(self, h):
        """
        Get the x, y, z coordinates from height above sea level.

        Parameters
        ----------
        h : float or array_like.

        Returns
        -------
        x, y, z : float or array_like.
        """
        h = np.array(h)
        z = h - self.atmosphere.h0
        x = self.x0 + z * self.ux / self.uz
        y = self.y0 + z * self.uy / self.uz
        return 1.*x, 1.*y, 1.*z

    def z_to_t(self, z):
        """
        Get the travel time from height above ground level.

        Parameters
        ----------
        z : float or array_like.

        Returns
        -------
        t : float or array_like.
        """
        return (self.z_top - z) / self.uz / 0.2998

    def X_to_xyz(self, X):
        """
        Get the x, y, z coordinates from slant depth.

        Parameters
        ----------
        X : float or array_like.

        Returns
        -------
        x, y, z : float or array_like.
        """
        Xv = X * self.uz
        h = self.atmosphere.Xv_to_h(Xv)
        return self.h_to_xyz(h)

    def Projection(self, telescope):
        """
        Obtain the coordinates of a shower track relative to the telescope
        position in both zenith and camera projection and determine the
        fraction of the track within the telescope field of view.

        Parameters
        ----------
        telescope : Telescope object.

        Returns
        -------
        Projection object.

        See also
        --------
        Projection.show
        """
        return sm.Projection(telescope, self)

    def show_projection(self, telescope, axes=True, max_theta=30.,
                        X_mark=None):
        """
        Obtain the polar coordinates of a shower track relative to a telescope
        position in both horizontal and FoV coordinates systems and determine
        the fraction of the track within the telescope field of view.
        In addition, show the projection of the shower track as viewed by the
        telescope.

        Parameters
        ----------
        telescope : Telescope object.
        axes : bool, default True
            Show the axes of both coordinate systems of reference.
        max_theta : float
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure. If None, no mark is included.

        Returns
        -------
        Projection object.
        (ax1, ax2) : PolarAxesSubplot objects.

        See also
        --------
        Projection.show
        """
        projection = sm.Projection(telescope, self)
        from ._tools import show_projection
        return projection, (show_projection(projection, None, False, axes,
                                            max_theta, X_mark))

    def show_geometry2D(self, observatory, x_min=-1., x_max=1., y_min=-1,
                        y_max=1., X_mark=None, tel_index=False):
        """
        Show the shower track together with the telescope positions in a
        2D plot.

        Parameters
        ----------
        x_min : float
            Lower limit of the coordinate x in km.
        x_max : float
            Upper limit of the coordinate x in km.
        y_min : float
            Lower limit of the coordinate y in km.
        y_max : float
            Upper limit of the coordinate y in km.
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure. If None, no mark is included.
        tel_index : bool, default False
            Show the telescope indexes together the telescope
            position points.

        Returns
        -------
        AxesSubplot object.
        """
        from ._tools import show_geometry
        return show_geometry(self, observatory, '2d', x_min, x_max, y_min,
                             y_max, X_mark, False, False, tel_index, False,
                             False)

    def show_geometry3D(self, observatory, x_min=-1., x_max=1., y_min=-1,
                        y_max=1., X_mark=None, xy_proj=True, pointing=False):
        """
        Show the shower track together with the telescope positions in a
        3D plot.

        Parameters
        ----------
        x_min : float
            Lower limit of the coordinate x in km.
        x_max : float
            Upper limit of the coordinate x in km.
        y_min : float
            Lower limit of the coordinate y in km.
        y_max : float
            Upper limit of the coordinate y in km.
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure, default to X_max. If X_mark is set to None,
            no mark is included.
        xy_proj : bool, default True
            Show the xy projection of the shower track.
        pointing : book, default False
            Show the telescope axes.

        Returns
        -------
        Axes3DSubplot object.
        """
        from ._tools import show_geometry
        return show_geometry(self, observatory, '3d', x_min, x_max, y_min,
                             y_max, X_mark, False, False, False, xy_proj,
                             pointing)

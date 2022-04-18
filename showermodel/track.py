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


# Class #######################################################################
class Track(pd.DataFrame):
    """
    DataFrame containing a linear-shower track discretization.

    Use sm.Track() to construct a default Track object.

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
    x0, y0 : float
        East and north coordinates in km of shower impact point at ground.
    xi, yi, zi : float, default None
        East, north and height coordinates in km of the first interaction point
        of the shower. If given, x0 and y0 are ignored.
    atmosphere : Atmosphere
        If None, a new Atmosphere object is generated.
    **kwargs : {h0, h_top, N_steps, model}
        Options to construct the new Atmosphere object when atm==None.
        If None, the default Atmosphere object is used.

    Attributes
    ----------
    x : float
        Column 0, east coordinate in km.
    y : float
        Column 1, north coordinate in km.
    z : float
        Column 2, height in km from ground level.
    t : float
        Column 3, travel time in microseconds. t=0 at the first interaction
        point. The shower is assumed to propagates with the speed of light.
    atmosphere : Atmosphere
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    ux, uy, uz : float
        Coordinates of a unit vector pointing to the source position
        (antiparallel to the shower propagation vector).
    vx, vy, vz : float
        Coordinates of a unit vector perpendicular to the shower axis and
        parallel to horizontal plane.
    wx, wy, wz : float
        Coordinates of a unit vector perpendicular to both u and v.
    x0, y0, z0 : float or None
        Coordinates in km of shower impact point at ground (z0=0).
        Set to None for ascending showers beginning at zi>0.
    xi, yi, zi : float
        Coordinates in km of the first interaction point of the shower.
    t_total : float
        Total travel time in microseconds.
    x_top, y_top, z_top : float or None
        Coordinates in km of shower at the top of the atmosphere.
        Set to None for descending showers beginning at zi<z_top.
    dl : float
        Size in km of discretization step along the shower axis.

    Methods
    -------
    h_to_xyz()
        Get the spatial coordinates from height above sea level.
    z_to_t()
        Get travel time mass density from height.
    Xv_to_xyz()
        Get the spatial coordinates from vertical depth.
    X_to_xyz()
        Get the spatial coordinates from travel depth.
    Projection()
        Make a Projection object containing the coordinates of a
        shower track relative to a telescope position.
    show_projection()
        Make a Projection object and show it.

    See also
    --------
    Shower : Make a discretization of a shower.
    """
    def __init__(self, theta=_theta, alt=None, az=_az, x0=_x0, y0=_y0,
                 xi=_x0, yi=_y0, zi=None, atmosphere=None, **kwargs):
        super().__init__(columns=['x', 'y', 'z', 't'])
        _track(self, theta, alt, az, x0, y0, xi, yi, zi, atmosphere, **kwargs)

    def h_to_xyz(self, h):
        """
        Get the x, y, z coordinates from height above sea level.

        Parameters
        ----------
        h : float or array_like

        Returns
        -------
        x, y, z : float, array_like or None
        """
        h = np.array(h)
        z = h - self.atmosphere.h0
        dist = (self.zi - z) / self.uz
        x = self.xi - dist * self.ux
        y = self.yi - dist * self.uy
        try: # for dist being a float
            if dist<0.:
                return None, None, None
            else:
                return 1.*x, 1.*y, 1.*z
        except Exception: # for dist being an array
            x[dist<0.] = None
            y[dist<0.] = None
            z[dist<0.] = None
            return x, y, z

    def z_to_t(self, z):
        """
        Get the travel time from height above ground level.

        Parameters
        ----------
        z : float or array_like

        Returns
        -------
        t : float, array_like or None
        """
        z = np.array(z)
        t = (self.zi - z) /self.uz / 0.2998
        try: # for t being a float
            if t<0.:
                return None, None, None
            else:
                return 1.*t
        except Exception: # for t being an array
            t[t<0.] = None
            return t

    def Xv_to_xyz(self, Xv):
        """
        Get the x, y, z coordinates from vertical depth.

        Parameters
        ----------
        Xv : float

        Returns
        -------
        x, y, z : float or None
        """
        h = self.atmosphere.Xv_to_h(Xv)
        if h is None:
            return None, None, None
        else:
            return self.h_to_xyz(h)

    def X_to_xyz(self, X):
        """
        Get the x, y, z coordinates from travel depth.

        Parameters
        ----------
        X : float

        Returns
        -------
        x, y, z : float or None
        """
        if self.uz>0. and self.z_top is not None:
            # Descending shower from the top of the atmosphere
            Xv = X * self.uz
        else: # Ascending or descending from zi
            hi = self.zi + self.atmosphere.h0
            Xv_i = self.atmosphere.h_to_Xv(hi)
            Xv = Xv_i + X * self.uz
        h = self.atmosphere.Xv_to_h(Xv)
        if h is None:
            return None, None, None
        else:
            return self.h_to_xyz(h)

    def Projection(self, telescope):
        """
        Obtain the coordinates of a shower track relative to the telescope
        position in both zenith and camera projection and determine the
        fraction of the track within the telescope field of view.

        Parameters
        ----------
        telescope : Telescope

        Returns
        -------
        projection : Projection

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
        telescope : Telescope
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
        projection : Projection
        (ax1, ax2) : PolarAxesSubplot

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
        ax : AxesSubplot
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
        pointing : bool, default False
            Show the telescope axes.

        Returns
        -------
        ax : Axes3DSubplot
        """
        from ._tools import show_geometry
        return show_geometry(self, observatory, '3d', x_min, x_max, y_min,
                             y_max, X_mark, False, False, False, xy_proj,
                             pointing)


# Constructor #################################################################
def _track(track, theta, alt, az, x0, y0, xi, yi, zi, atmosphere, **kwargs):
    """
    Constructor of Track class.

    Parameters
    ----------
    track : Track
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apperent position of the source. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0, y0 : float
    xi, yi, zi : float
    atmosphere : Atmosphere
        If None, a new Atmosphere object is generated.
    **kwargs : {h0, h_top, N_steps, model}
        Options to construct the new Atmosphere object when atm==None.
        If None, the default Atmosphere object is used.
    """
    if isinstance(atmosphere, sm.Atmosphere):
        pass
    elif atmosphere is None:
        atmosphere = sm.Atmosphere(**kwargs)
    else:
        raise ValueError('The input atmosphere is not valid.')
    track.atmosphere = atmosphere
    z_top = atmosphere.h_top - atmosphere.h0

    if alt is None:
        if theta<0. or theta>180. or theta==90.:
            raise ValueError('The input theta value is not valid.')
        alt = 90. - theta
    else:
        if alt<-90. or alt>=90.:
            raise ValueError('The input alt value is not valid.')
        theta = 90. - alt
    
    # The input parameters along with some geometric parameters are also
    # included as atributes of the DataFrame. The angles are stored in degrees
    track.theta = theta
    track.alt = alt
    if theta==180.:
        theta = math.pi
        cos_theta = -1.
        sin_theta = 0.
    else:
        theta = math.radians(theta)
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
    track.az = az
    az = math.radians(az)
    cos_az = math.cos(az)
    sin_az = math.sin(az)

    # Coordinates of the unit vector pointing at the arrival shower direction
    # opposite to the shower propogation vector
    track.ux = sin_theta * sin_az
    track.uy = sin_theta * cos_az
    track.uz = cos_theta # uz<0 for ascending showers

    # Coordinates of a unit vector perpendicular to u and parallel to
    # horizontal xy plane
    track.vx = cos_az
    track.vy = -sin_az
    track.vz = 0.

    # Coordinates of the unit vector perpendicular to both u and v.
    track.wx = cos_theta * sin_az
    track.wy = cos_theta * cos_az
    track.wz = -sin_theta

    # Distance in km travelled through one atmospheric slice
    track.dl = atmosphere.h_step / abs(track.uz)

    z = atmosphere.h - atmosphere.h0
    if zi is None:
        track.x0 = x0
        track.y0 = y0
        track.z0 = 0.
        dist_top = z_top / track.uz
        track.z_top = z_top
        track.x_top = x0 + dist_top * track.ux
        track.y_top = y0 + dist_top * track.uy

        # Total travel time in us, where t=0 corresponds to the moment when the
        # shower begins
        track.t_total = abs(dist_top) / 0.2998

        # Coordinates along the shower track
        track.z = z
        dist = z / track.uz
        track.x = x0 + dist * track.ux
        track.y = y0 + dist * track.uy

        if track.uz>0.: # descending shower from the top of the atmosphere
            track.zi = track.z_top
            track.xi = track.x_top
            track.yi = track.y_top
            # Travel time
            track.t = (dist_top - dist) / 0.2998
        else: # ascending shower from ground
            track.zi = track.z0
            track.xi = track.x0
            track.yi = track.y0
            # Travel time
            track.t = abs(dist) / 0.2998

    else:
        track.xi = xi
        track.yi = yi
        track.zi = zi

        if track.uz>0: # descending shower from arbitrary initial height
            if zi==z_top:
                track.z_top = zi
                track.x_top = xi
                track.y_top = yi
            else:
                track.z_top = None
                track.x_top = None
                track.y_top = None
            points = atmosphere[z<=zi].index
            track.z0 = 0.
            dist = zi / track.uz
            track.x0 = xi - dist * track.ux
            track.y0 = yi - dist * track.uy

            # Total travel time in us, where t=0 corresponds to the moment when
            # the shower begins
            track.t_total = zi / track.uz / 0.2998

        else: # ascending shower from arbitrary initial height
            if zi==0.:
                track.z0 = 0.
                track.x0 = xi
                track.y0 = yi
            else:
                track.z0 = None
                track.x0 = None
                track.y0 = None
            points = atmosphere[z>=zi].index
            track.z_top = z_top
            dist = (zi - z_top) / track.uz
            track.x_top = xi - dist * track.ux
            track.y_top = yi - dist * track.uy
            
            # Total travel time in us, where t=0 corresponds to the moment when
            # the shower begins
            track.t_total = dist / 0.2998

        # Coordinates along the shower track
        track.z = z[points]
        dist = (zi - track.z) / track.uz
        track.x = xi - dist * track.ux
        track.y = yi - dist * track.uy
        # Travel time
        track.t = dist / 0.2998
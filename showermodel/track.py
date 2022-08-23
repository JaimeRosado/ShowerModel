# coding: utf-8

import numpy as np
import pandas as pd
import showermodel as sm
import showermodel.constants as ct
import warnings
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)


# Default values for Track
_Track__theta = ct.config['Shower']['theta']
_Track__alt = ct.config['Shower'].get('alt') # optional parameter
_Track__az = ct.config['Shower']['az']
_Track__x0 = ct.config['Shower']['x0']
_Track__y0 = ct.config['Shower']['y0']
_Track__xi = ct.config['Shower']['xi']
_Track__yi = ct.config['Shower']['yi']
_Track__zi = ct.config['Shower'].get('zi') # optional parameter
_Track__h0 = ct.config['Atmosphere']['h0']
_Track__h_top = ct.config['Atmosphere'].get('h_top') # optional parameter
_Track__N_steps = ct.config['Atmosphere']['N_steps']
_Track__atm_model = ct.config['Atmosphere']['atm_model']
_Track__rho_w_sl = ct.config['Atmosphere']['rho_w_sl']
_Track__h_scale = ct.config['Atmosphere']['h_scale']

# Class #######################################################################
class Track(pd.DataFrame):
    """
    DataFrame containing a linear-shower track discretization.

    Use sm.Track() to construct a default Track object.

    Parameters
    ----------
    theta : float, default 0
        Zenith angle in degrees of the apparent position of the source.
<<<<<<< Updated upstream
    alt : float
=======
    alt : float, default None
>>>>>>> Stashed changes
        Altitude in degrees of the apparent position of the source. If None,
        theta is used. If given, theta is overwritten.
    az : float, default 0
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0 : float, default 0
        East coordinate in km of shower impact point on ground.
    y0 : float, default 0
        North coordinate in km of shower impact point on ground.
    xi : float, default 0
        East coordinate in km of the first interaction point of the shower.
        If zi==None, xi and yi are ignored and the shower impacts at (x0, y0)
        on ground. If zi is given, x0 and y0 are ignored and the shower starts
        at (xi,yi,zi).
    yi : float, default 0
        North coordinate in km of the first interaction point of the shower.
        If zi==None, xi and yi are ignored and the shower impacts at (x0, y0)
        on ground. If zi is given, x0 and y0 are ignored and the shower starts
        at (xi,yi,zi).
    zi : float, default None
        Height in km of the first interaction point of the shower. If zi==None,
        xi and yi are ignored and the shower impacts at (x0, y0) on ground.
        If zi is given, x0 and y0 are ignored and the shower starts at
        (xi,yi,zi).
    atmosphere : Atmosphere, default None
        Atmosphere object to be used. If None, a new Atmosphere object
        is generated. If given, h0, h_top, N_steps, atm_model, rho_w_sl
        and h_scale are ignored.
    h0 : float, default 0
        Ground level in km above sea level for the atmosphere
        discretization to be generated when atmosphere==None.
    h_top : float or None, default None
        Upper limit in km above sea level for the atmosphere
        discretization to be generated when atmosphere==None. If h_top
        is None, the top level of the selected atmospheric model is
        taken.
    N_steps : int, default 550
        Number of discretization steps for the atmosphere discretization
        to be generated when atmosphere==None.
    atm_model : int or DataFrame, default 1
        Atmospheric model used when atmosphere==None. If an int value
        is given, atm_model is searched from either CORSIKA atmospheric
        models (from 1 to 29) or a file named atm_models.toml in the
        working directory containing user-defined models. If a
        DataFrame is given, it should have two columns, one labelled as
        h with height in km and other labelled as X_vert or P,
        depending on whether vertical depth in g/cm^2 or pressure in
        hPa is given.
    rho_w_sl : float, default 7.5e-6
        Water-vapor density in g/cm^3 at sea level to calculate a
        simple exponential profile of water-vapor when
        atmosphere==None. Set to zero if dry air is assumed.
    h_scale : float, default 2.0
        Scale height in km to be used in the water-vapor exponential
        profile when atmospere==None.

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
        point. The shower is assumed to propagate with the speed of light.
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
    def __init__(self, theta=__theta, alt=__alt, az=__az, x0=__x0, y0=__y0,
                 xi=__xi, yi=__y0, zi=__zi, atmosphere=None, h0=__h0,
                 h_top=__h_top, N_steps=__N_steps, atm_model=__atm_model,
                 rho_w_sl=__rho_w_sl, h_scale=__h_scale):
        super().__init__(columns=['x', 'y', 'z', 't'])
        _track(self, theta, alt, az, x0, y0, xi, yi, zi, atmosphere, h0,
               h_top, N_steps, atm_model, rho_w_sl, h_scale)
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
        t = (self.zi - z) /self.uz / ct.c_km_us
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
        telescope : Telescope, mandatory

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
        telescope : Telescope, mandatory
        axes : bool, default True
            Show the axes of both coordinate systems of reference.
        max_theta : float, default 30
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float, default None
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
        x_min : float, default -1
            Lower limit of the coordinate x in km.
        x_max : float, default 1
            Upper limit of the coordinate x in km.
        y_min : float, default -1
            Lower limit of the coordinate y in km.
        y_max : float, default 1
            Upper limit of the coordinate y in km.
        X_mark : float, default None
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
        x_min : float, default -1
            Lower limit of the coordinate x in km.
        x_max : float, default 1
            Upper limit of the coordinate x in km.
        y_min : float, default -1
            Lower limit of the coordinate y in km.
        y_max : float, default 1
            Upper limit of the coordinate y in km.
        X_mark : float, default None
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure, default to X_max. If None, no mark is
            included.
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
def _track(track, theta, alt, az, x0, y0, xi, yi, zi, atmosphere, h0, h_top,
           N_steps, atm_model, rho_w_sl, h_scale):
    """
    Constructor of Track class.

    Parameters
    ----------
    track : Track
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0, y0 : float
    xi, yi, zi : float
    atmosphere : Atmosphere, default None
        Atmosphere object to be used. If None, a new Atmosphere object is
        generated.
    h0 : float, default 0.0
        Ground level in km above sea level of the Atmosphere object to be
        generated when atmosphere==None.
    h_top : float, default 112.8292
        Top level of the atmosphere in km above sea level of the Atmosphere
        object to be generated when atmosphere==None.
    N_steps : int, default 550
        Number of discretization steps of the Atmosphere object to be
        generated when atmosphere==None.
    atm_model : int or DataFrame
        Atmospheric model assuming dry air.
    rho_w_sl : float
        Water-vapor density in g/cm^3 at sea level to calculate a simple
        exponential profile of water-vapor. Set to zero if dry air is assumed.
    h_scale : float
        Scale height in km to be used in the water-vapor exponential profile.
    """
    if isinstance(atmosphere, sm.Atmosphere):
        pass
    elif atmosphere is None:
        atmosphere = sm.Atmosphere(h0, h_top, N_steps, atm_model,
                                   rho_w_sl, h_scale)
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
    # included as attributes of the DataFrame. The angles are stored in degrees
    track.theta = theta
    track.alt = alt
    if theta==180.:
        theta = ct.pi
        cos_theta = -1.
        sin_theta = 0.
    else:
        theta = np.radians(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
    track.az = az
    az = np.radians(az)
    cos_az = np.cos(az)
    sin_az = np.sin(az)

    # Coordinates of the unit vector pointing at the arrival shower direction
    # opposite to the shower propagation vector
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
        track.t_total = abs(dist_top) / ct.c_km_us

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
            track.t = (dist_top - dist) / ct.c_km_us
        else: # ascending shower from ground
            track.zi = track.z0
            track.xi = track.x0
            track.yi = track.y0
            # Travel time
            track.t = abs(dist) / ct.c_km_us

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
            track.t_total = zi / track.uz / ct.c_km_us

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
            track.t_total = dist / ct.c_km_us

        # Coordinates along the shower track
        track.z = z[points]
        dist = (zi - track.z) / track.uz
        track.x = xi - dist * track.ux
        track.y = yi - dist * track.uy
        # Travel time
<<<<<<< Updated upstream
        track.t = dist / 0.2998
=======
        track.t = dist / ct.c_km_us
>>>>>>> Stashed changes

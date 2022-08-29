# coding: utf-8

import numpy as np
import pandas as pd
import showermodel as sm
import showermodel.constants as ct

# Default parameters for Telescope
_Telescope__tel_type = ct.config['Telescope'].get('tel_type') # optional parameter
_Telescope__x = ct.config['Telescope']['x']
_Telescope__y = ct.config['Telescope']['y']
_Telescope__z = ct.config['Telescope']['z']
_Telescope__theta = ct.config['Telescope']['theta']
_Telescope__alt = ct.config['Telescope'].get('alt') # optional parameter
_Telescope__az = ct.config['Telescope']['az']
_Telescope__tel_type = ct.config['Telescope']['tel_type']


# Class #######################################################################
class Telescope:
    """
    Object containing the characteristics of a Cherenkov/fluorescence
    telescope.

    Parameters
    ----------
    tel_type : str, default 'generic'
        Type of telescope to be used.
    x : float, default 0
        East coordinate of the telescope in km.
    y : float, default 0
        North coordinate of the telescope in km.
    z : float, default 0
        Height of the telescope in km above ground level.
    theta : float, default 0
        Zenith angle in degrees of the telescope pointing direction.
    alt : float, default None
        Altitude in degrees of the telescope pointing direction.
        If None, theta is used. If given, theta is overwritten.
    az : float, default 0
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing direction.
    apert : float, default 10.
        Angular diameter in degrees of the telescope field of view.
    area : float, default 100.
        Detection area in m^2 (e.g., mirror area of an IACT).
    N_pix : int, default 1500
        Number of camera pixels.
    int_time : float, default 0.01
        Integration time in microseconds of camera frames.
    wvl : array_like, default None
        Wavelength interval in nm with constant discretization step.
        If None, it is calculated from wvl_ini, wvl_fin, wvl_step.
    wvl_ini : float, default 290
        Initial wavelength in nm of the interval where the efficiency is
        non zero.
    wvl_fin : float, default 430
        Final wavelength in nm of the interval where the efficiency is
        non zero.
    wvl_step : float, default 3
        Discretization step in nm of the interval where the efficiency is
        non zero.
    eff : float or array_like, default 1
        Detection efficiency in decimal fraction. If a float value is given,
        efficiency is assumed to be constant within the wavelength interval
        [wvl_ini, wvl_fin]. If a a list of efficiency values is given,
        it should match wvl.
    eff_fluo : array_like or None, default None
        Detection efficiency at the 57 bands considered in the fluorescence
        model. If None, values are interpolated from eff.

    Attributes
    ----------
    tel_type : str
        Name given to the telescope. Default to None.
    apert : float
        Angular diameter in degrees of the telescope field of view.
    area : float
        Detection area in m^2 (e.g., mirror area of an IACT).
    N_pix : int
        Number of camera pixels. Default to 1500.
    int_time : float
        Integration time in microseconds of camera frames.
    sol_angle : float
        Telescope field of view in steradians.
    sol_angle_pix : float
        Pixel field of view in steradians.
    apert_pix : float
        Angular diameter in degrees of the pixel FoV.
    wvl_ini : float
        Initial wavelength in nm of the detection efficiency data.
    wvl_fin : float
        Final wavelength in nm of the detection efficiency data.
    wvl_step : float
        Step size in nm of the detection efficiency data.
    wvl : ndarray
        Array containing the range of wavelengths in nm defined by
        wvl_ini, wvl_fin and wvl_step
    eff : ndarray
        Array containing the detection efficiency values.
    wvl_fluo : ndarray
        Array containing the wavelengths of fluorescence bands within
        the wavelength interval.
    eff_fluo : ndarray
        Array containing the detection efficiency values at the fluorescence
        bands within the wavelength interval.
    x : float
        East coordinate of the telescope in km.
    y : float
        North coordinate of the telescope in km.
    z : float
        Height of the telescope in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing direction.
    alt : float
        Altitude in degrees of the telescope pointing direction.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing direction.
    ux : float
        x coordinate of a unit vector parallel to the telescope
        pointing direction.
    uy : float
        y coordinate of a unit vector parallel to the telescope
        pointing direction.
    uz : float
        z coordinate of a unit vector parallel to the telescope
        pointing direction.

    Methods
    -------
    copy()
        Copy the Telescope object, but with optional changes.
    hor_to_FoV()
        Convert cartesian coordinates from horizontal system to FoV system.
    FoV_to_hor()
        Convert cartesian coordinates from FoV system to horizontal system.
    theta_phi_to_alt_az()
        Convert FoV coordinates theta/phi to horizontal coordinates alt/az.
    alt_az_to_theta_phi()
        Convert horizontal coordinates alt/az to FoV coordinates theta/phi.
    spherical()
        Calculate the spherical coordinates in both horizontal and FoV systems.
    abs_to_rel()
        Calculate the x, y, z coordinates relative to the telescope
        position from the 'absolute' x, y, z coordinates.
    distance()
        Calculate the distance in km between the point x, y, z
        ('absolute' coordinates) and the telescope position.

    See also
    --------
    IACT : IACT class, daughter of Telescope class.
    GridElement : GridElement class, daughter of Telescope class.
    Observatory : List of telescopes.
    """

    # Methods #################################################################
    # __name is replaced by _Telescope__name
    def __init__(self, x=__x, y=__y, z=__z, theta=__theta, alt=__alt, az=__az,
                 tel_type=__tel_type, **kwargs):
        # Default values for tel_type
        tel = ct.tel_data.get(str(tel_type))
        if tel is None:
            raise ValueError("This tel_type is not implemented.")
        self.tel_type = tel_type
        __apert = tel['apert']
        __area = tel['area']
        __N_pix = tel['N_pix']
        __int_time = tel['int_time']
        __wvl = tel.get('wvl') # optional parameter
        __wvl_ini = tel['wvl_ini']
        __wvl_fin = tel['wvl_fin']
        __wvl_step = tel['wvl_step']
        __eff = tel['eff']
        __eff_fluo = tel.get('eff_fluo') # optional parameter

        # Load telescope parameters. Default to those stored for tel_type
        apert = kwargs.get('apert', __apert)
        area = kwargs.get('area', __area)
        N_pix = kwargs.get('N_pix', __N_pix)
        int_time = kwargs.get('int_time', __int_time)
        wvl = kwargs.get('wvl', __wvl)
        wvl_ini = kwargs.get('wvl_ini', __wvl_ini)
        wvl_fin = kwargs.get('wvl_fin', __wvl_fin)
        wvl_step = kwargs.get('wvl_step', __wvl_step)
        eff = kwargs.get('eff', __eff)
        eff_fluo = kwargs.get('eff_fluo', __eff_fluo)
        _telescope(self, x, y, z, theta, alt, az, apert, area, N_pix,
               int_time, wvl, wvl_ini, wvl_fin, wvl_step, eff, eff_fluo)


    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, new_theta):
        _pointing(self, new_theta, None, self._az)

    @property
    def alt(self):
        return self._alt

    @alt.setter
    def alt(self, new_alt):
        _pointing(self, None, new_alt, self._az)

    @property
    def az(self):
        return self._az

    @az.setter
    def az(self, new_az):
        _pointing(self, self._theta, None, new_az)

    @property
    def apert(self):
        return self._apert

    @apert.setter
    def apert(self, new_apert):
        _solid_angle(self, new_apert, self._N_pix)

    @property
    def N_pix(self):
        return self._N_pix

    @N_pix.setter
    def N_pix(self, new_N_pix):
        _solid_angle(self, self._apert, new_N_pix)

    @property
    def sol_angle(self):
        return self._sol_angle

    @property
    def sol_angle_pix(self):
        return self._sol_angle_pix

    @property
    def apert_pix(self):
        return self._apert_pix

    @property
    def ux(self):
        return self._ux

    @property
    def uy(self):
        return self._uy

    @property
    def uz(self):
        return self._uz

    @property
    def wvl(self):
        return self._wvl

    @property
    def wvl_ini(self):
        return self._wvl_ini

    @property
    def wvl_fin(self):
        return self._wvl_fin

    @property
    def wvl_step(self):
        return self._wvl_step

    @property
    def eff(self):
        return self._eff

    @property
    def wvl_fluo(self):
        return self._wvl_fluo

    @property
    def eff_fluo(self):
        return self._eff_fluo

    def copy(self, **kwargs):
        """
        Copy a Telescope object, but with optional changes.

        Parameters
        ----------
        **kwargs : {x, y, z, ...}
            Optional changes to the original telescope attributes.

        Returns
        -------
        telescope : Telescope
        """
        kwargs['x'] = kwargs.get('x', self.x)
        kwargs['y'] = kwargs.get('y', self.y)
        kwargs['z'] = kwargs.get('z', self.z)
        # If 'alt' in kwargs and != None, theta is not used
        kwargs['theta'] = kwargs.get('theta', self.theta)
        kwargs['az'] = kwargs.get('az', self.az)
        kwargs['tel_type'] = kwargs.get('tel_type', self.tel_type)
        kwargs['apert'] = kwargs.get('apert', self.apert)
        kwargs['area'] = kwargs.get('area', self.area)
        kwargs['N_pix'] = kwargs.get('N_pix', self.N_pix)
        kwargs['int_time'] = kwargs.get('int_time', self.int_time)
        kwargs['wvl'] = kwargs.get('wvl', 1.*self.wvl)
        kwargs['wvl_ini'] = kwargs.get('wvl_ini', self.wvl_ini)
        kwargs['wvl_fin'] = kwargs.get('wvl_fin', self.wvl_fin)
        kwargs['wvl_step'] = kwargs.get('wvl_step', self.wvl_fin)
        kwargs['eff'] = kwargs.get('eff', 1.*self.eff)
        kwargs['eff_fluo'] = kwargs.get('eff_fluo', 1.*self.eff_fluo)

        telescope_c = Telescope(**kwargs)
        return telescope_c

    def alt_az_to_theta_phi(self, alt, az):
        """
        Convert polar horizontal coordinates alt, az to FoV coordinates
        theta, phi.

        Parameters
        ----------
        alt, az : float or array_like

        Returns
        -------
        theta, phi : float or array_like

        See also
        --------
        Telescope.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Telescope.theta_phi_to_alt_az : Convert FoV coordinates theta, phi to
            horizontal coordinates alt, az.
        """
        alt = np.radians(alt)
        az = np.radians(az)

        x_hor = np.cos(alt) * np.sin(az)
        y_hor = np.cos(alt) * np.cos(az)
        z_hor = np.sin(alt)

        x_FoV, y_FoV, z_FoV = self.hor_to_FoV(x_hor, y_hor, z_hor)

        theta = self.zr_to_theta(z_FoV, 1.)
        phi = self.xy_to_phi(x_FoV, y_FoV) + self._phi_right - 360.
        try:  # For phi being a scalar
            phi = phi + 360. if phi < 0. else phi   # [0, 360)
        except Exception:  # For phi being an array
            phi[phi < 0.] = phi[phi < 0.] + 360.   # [0, 360)

        return theta, phi

    def hor_to_FoV(self, x_hor, y_hor, z_hor):
        """
        Convert cartesian coordinates from horizontal system to FoV system.

        In the FoV coordinates system, x_FoV grows in the right-hand direction,
        y_FoV grows downward and z_FoV grows toward the pointing direction from
        the telescope point of view.

        Parameters
        ----------
        x_hor, y_hor, z_hor : float or array_like

        Returns
        -------
        x_FoV, y_FoV, z_FoV : float or array_like

        See also
        --------
        Telescope.FoV_to_hor : Convert cartesian coordinates from FoV system to
            horizontal system.
        Telescope.alt_az_to_theta_phi : Convert horizontal coordinates alt, az
            to FoV coordinates theta, phi.
        """
        sin_theta = self._sin_theta
        cos_theta = self._cos_theta
        sin_az = self._sin_az
        cos_az = self._cos_az

        x_FoV = cos_az * x_hor - sin_az * y_hor
        y_FoV = (cos_theta * sin_az * x_hor + cos_theta * cos_az * y_hor
                 - sin_theta * z_hor)
        z_FoV = (sin_theta * sin_az * x_hor + sin_theta * cos_az * y_hor
                 + cos_theta * z_hor)

        return x_FoV, y_FoV, z_FoV

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
        Telescope.FoV_to_hor : Convert cartesian coordinates from FoV system to
            horizontal system.
        Telescope.alt_az_to_theta_phi : Convert horizontal coordinates alt, az
            to FoV coordinates theta, phi.
        """
        theta = np.radians(theta)
        phi = np.radians(phi - self._phi_right + 360.)

        x_FoV = np.sin(theta) * np.cos(phi)
        y_FoV = np.sin(theta) * np.sin(phi)
        z_FoV = np.cos(theta)

        x_hor, y_hor, z_hor = self.FoV_to_hor(x_FoV, y_FoV, z_FoV)

        alt = 90. - self.zr_to_theta(z_hor, 1.)  # [-90, 90]
        az = 90. - self.xy_to_phi(x_hor, y_hor)  # (-180, 180]
        try:  # For az being a scalar
            az = az + 360. if az < 0. else az  # [0, 360)
        except Exception:  # For az being an array
            az[az < 0.] = az[az < 0.] + 360.  # [0, 360)
        return alt, az

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
        x_hor, y_hor, z_hor : float or array_like

        See also
        --------
        Telescope.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Telescope.theta_phi_to_alt_az : Convert FoV coordinates theta, phi to
            horizontal coordinates alt, az.
        """
        sin_theta = self._sin_theta
        cos_theta = self._cos_theta
        sin_az = self._sin_az
        cos_az = self._cos_az

        x_hor = (cos_az * x_FoV + cos_theta * sin_az * y_FoV + sin_theta
                 * sin_az * z_FoV)
        y_hor = (-sin_az * x_FoV + cos_theta * cos_az * y_FoV + sin_theta
                 * cos_az * z_FoV)
        z_hor = -sin_theta * y_FoV + cos_theta * z_FoV

        return x_hor, y_hor, z_hor

    def abs_to_rel(self, x, y, z):
        """
        Calculate the x, y, z coordinates relative to the telescope position
        from the 'absolute' x, y, z coordinates.
        """
        return x - self.x, y - self.y, z - self.z

    def distance(self, x, y, z):
        """
        Calculate the distance in km between the point x, y, z
        ('absolute' coordinates) and the telescope position.

        Parameters
        ----------
        x, y, z : float or array_like

        Returns
        -------
        distance: float or array_like
        """
        x_rel, y_rel, z_rel = self.abs_to_rel(x, y, z)
        return np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

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
        x_hor, y_hor, z_hor = self.abs_to_rel(x, y, z)
        distance = np.sqrt(x_hor**2 + y_hor**2 + z_hor**2)

        alt = 90. - self.zr_to_theta(z_hor, distance)  # [-90, 90]
        az = 90. - self.xy_to_phi(x_hor, y_hor)  # (-180, 180]
        try:  # For az being a scalar
            az = az + 360. if az < 0. else az  # [0, 360)
        except Exception:  # For az being an array
            az[az < 0.] = az[az < 0.] + 360.  # [0, 360)

        x_FoV, y_FoV, z_FoV = self.hor_to_FoV(x_hor, y_hor, z_hor)
        theta = self.zr_to_theta(z_FoV, distance)  # [0, 180]
        phi = self.xy_to_phi(x_FoV, y_FoV) + self._phi_right - 360.
        # [-90, 270) - (0, 360] = (-450, 270)

        try:  # For phi being a scalar
            phi = phi + 360. if phi < 0. else phi  # [0, 360)
        except Exception:  # For phi being an array
            phi[phi < 0.] = phi[phi < 0.] + 360.  # [0, 360)
        return distance, alt, az, theta, phi

    def zr_to_theta(self, z, r):
        """
        Calculate the angle theta in degrees [0, 180] of a vector with vertical
        projection z and modulus r, where theta is defined from the z axis
        
        Parameters
        ----------
        z, r : float or array_like
        """
        from ._tools import zr_to_theta
        return zr_to_theta(z, r)

    def xy_to_phi(self, x, y):
        """
        Calculate the angle phi in degrees [-90, 270) of the xy projection of a
        vector, where phi is defined from the x axis towards the y axis
        (anticlockwise)
        
        Parameters
        ----------
        x, y : float or array_like
        """
        from ._tools import xy_to_phi
        return xy_to_phi(x, y)

    def Projection(self, track):
        """
        Obtain the coordinates of a shower track relative to the telescope
        position in both zenith and camera projection and determine the
        fraction of the track within the telescope field of view.

        Parameters
        ----------
        track : Track or Shower, mandatory

        Returns
        -------
        projection : Projection

        See also
        --------
        Projection.show
        """
        return sm.Projection(self, track)

    def show_projection(self, track, axes=True, max_theta=30., X_mark=None):
        """
        Obtain the polar coordinates of a shower track relative to a telescope
        position in both horizontal and FoV coordinates systems and determine
        the fraction of the track within the telescope field of view.
        In addition, show the projection of the shower track as viewed by the
        telescope.

        Parameters
        ----------
        track : Track or Shower, mandatory
            Track object to be used.
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
        projection = sm.Projection(self, track)
        from ._tools import show_projection
        return projection, (show_projection(projection, None, False, axes,
                                            max_theta, X_mark))


# Constructor #################################################################
def _telescope(telescope, x, y, z, theta, alt, az, apert, area, N_pix,
               int_time, wvl, wvl_ini, wvl_fin, wvl_step, eff, eff_fluo):
    """
    Constructor of Telescope class and daughter classes.

    Parameters
    ----------
    telescope : Telescope
    x : float
        East coordinate of the telescope in km.
    y : float
        North coordinate of the telescope in km.
    z : float
        Height of the telescope in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing direction.
    alt : float
        Altitude in degrees of the telescope pointing direction. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing direction.
    apert : float
        Angular diameter in degrees of the telescope field of view.
    area : float
        Detection area in m^2 (e.g., mirror area of an IACT).
    N_pix : int
        Number of camera pixels.
    int_time : float
        Integration time in microseconds of camera frames.
    wvl : array_like or None
        Wavelength interval. If None, wvl is calculated from wvl_ini, wvl_fin,
        wvl_step.
    wvl_ini, wvl_fin, wvl_step : float
        Parameters defining the wavelength interval in nm where the efficiency
        is non zero when a float value is given to efficiency.
    eff : float or array_like
    eff_fluo : array_like or None
    """
    telescope.x = x
    telescope.y = y
    telescope.z = z

    _pointing(telescope, theta, alt, az)
    _solid_angle(telescope, apert, N_pix)
    telescope.area = area
    telescope.int_time = int_time
    _efficiency(telescope, wvl, wvl_ini, wvl_fin, wvl_step, eff, eff_fluo)

def _pointing(telescope, theta, alt, az):
    """
    Set pointing parameters.
    """
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    telescope._theta = theta
    telescope._alt = alt
    telescope._az = az

    telescope._sin_theta = np.sin(np.radians(theta))
    telescope._cos_theta = np.cos(np.radians(theta))
    telescope._sin_az = np.sin(np.radians(az))
    telescope._cos_az = np.cos(np.radians(az))

    telescope._ux = telescope._sin_theta * telescope._sin_az
    telescope._uy = telescope._sin_theta * telescope._cos_az
    telescope._uz = telescope._cos_theta

    # Coordinates of a point at 1km distance from the telescope in the north
    # direction
    x_north, y_north, z_north = telescope.hor_to_FoV(0., 1., 0.)
    # Position angle of the right-hand direction relative to north
    # direction
    phi_right = - telescope.xy_to_phi(x_north, y_north)  # (-270, 90]
    telescope._phi_right = phi_right + 360. if phi_right < 0. else phi_right
    # [0, 360)

def _solid_angle(telescope, apert, N_pix):
    """
    Set solid-angle parameters.
    """
    telescope._apert = apert
    telescope._N_pix = N_pix

    sol_angle = 2. * ct.pi*(1.-np.cos(np.radians(apert)/2.))  # str
    telescope._sol_angle = sol_angle
    sol_angle_pix = sol_angle / N_pix    # str
    telescope._sol_angle_pix = sol_angle_pix
    apert_pix = 2. * np.degrees(np.arccos(1.-sol_angle_pix/2./ct.pi))  # deg
    telescope._apert_pix = apert_pix
    # Delta_pix = np.sqrt(sol_angle_pix / 2.)
    # Delta_r = np.sqrt(sol_angle / 2 / ct.pi)

def _efficiency(telescope, wvl, wvl_ini, wvl_fin, wvl_step, eff, eff_fluo):
    """
    Set efficiency parameters.
    """
    if wvl is None:
        wvl = np.arange(wvl_ini, wvl_fin, wvl_step)
        telescope._wvl = wvl
    else: # wvl is given, so input wvl_ini, wvl_fin, wvl_step are ignored
        wvl = np.array(wvl)
        wvl_ini = wvl[0]
        wvl_step = wvl[1] - wvl_ini
        wvl_fin = wvl[-1] + wvl_step / 2.
        if not np.all(np.diff(wvl)==wvl_step):
            raise ValueError(
                "The wavelength discretization step must be constant.")
        telescope._wvl = wvl

    telescope._wvl_ini = wvl_ini
    telescope._wvl_fin = wvl_fin
    telescope._wvl_step = wvl_step
    wvl_fluo = np.array(ct.fluo_model['wvl'])
    telescope._wvl_fluo = wvl_fluo

    if isinstance(eff, float):
        # Constant efficiency is assumed
        telescope._eff = np.full_like(wvl, eff)
        eff_fluo = np.full_like(wvl_fluo, eff)
        telescope._eff_fluo = eff_fluo

    else: # eff array is given
        eff = np.array(eff)
        if len(wvl)!=len(eff):
            raise ValueError("Lengths of wvl and eff do not match.")
        telescope._eff = eff
        
        if eff_fluo is None: # Efficiency values are interpolated
            eff_fluo = np.interp(wvl_fluo, wvl, eff, left=0., right=0.)
        else: # eff_fluo is given
            eff_fluo = np.array(eff_fluo)
        telescope._eff_fluo = eff_fluo

# coding: utf-8

import math
import numpy as np
import pandas as pd
import showermodel as sm  # For projection

# Default values for telescope
_x = 0.  # km
_y = 0.  # km
_z = 0.  # km
_theta = 0.  # deg
_az = 0.  # deg
_tel_type = 'IACT'


# Constructor #################################################################
def Telescope(x=_x, y=_y, z=_z, theta=_theta, alt=None, az=_az,
              tel_type=_tel_type, efficiency=None, apert=None, area=None,
              N_pix=None, int_time=None):
    """
    Make a Telescope object with the specified characteristics.

    Parameters
    ----------
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
    tel_type : str
        Subclass of Telescope to be used, default to IACT. If None, the
        parent class Telescope is used. Presently only the IACT and GridElement
        subclasses are available. More subclasses to be implemented.
    efficiency : DataFrame
        If None, the default efficiency of the selected tel_type. If given,
        the DataFrame should have two columns with wavelength in nm
        (with constant discretization step) and efficiency (decimal fraction).
    apert : float
        Angular diameter in degrees of the telescope field of view.
    area : float
        Detection area in m^2 (e.g., mirror area of an IACT).
    N_pix : int
        Number of camera pixels.
    int_time : float
        Integration time in microseconds of camera frames.

    Returns
    -------
    telescope : Telescope object.

    See also
    --------
    _Telescope : Telescope class.
    Array25 : Make an array of 25 telescopes based on a layout of CTA.
    """
    if tel_type == 'IACT':
        telescope = _IACT()
    elif tel_type == 'GridElement':
        telescope = _GridElement()
    else:
        telescope = _Telescope()
        # If tel_type!=None -> new telescope type
        telescope.tel_type = tel_type

    telescope.x = x
    telescope.y = y
    telescope.z = z
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    telescope.theta = theta
    telescope.alt = alt
    telescope.az = az

    telescope.sin_theta = math.sin(math.radians(theta))
    telescope.cos_theta = math.cos(math.radians(theta))
    telescope.sin_az = math.sin(math.radians(az))
    telescope.cos_az = math.cos(math.radians(az))

    telescope.ux = telescope.sin_theta * telescope.sin_az
    telescope.uy = telescope.sin_theta * telescope.cos_az
    telescope.uz = telescope.cos_theta

    # Coordinates of a point at 1km distance from the telescope in the north
    # direction
    x_north, y_north, z_north = telescope.hor_to_FoV(0., 1., 0.)
    # Position angle of the the right-hand direction relative to north
    # direction
    phi_right = - telescope.xy_to_phi(x_north, y_north)  # (-270, 90]
    telescope.phi_right = phi_right + 360. if phi_right < 0. else phi_right
    # [0, 360)

    if (apert is not None) or (N_pix is not None):
        if apert is not None:
            telescope.apert = apert
            telescope.sol_angle = (
                2. * math.pi * (1.-math.cos(math.radians(telescope.apert)/2.)))
            # str

        if N_pix is not None:
            telescope.N_pix = N_pix

        telescope.sol_angle_pix = telescope.sol_angle / telescope.N_pix  # str
        telescope.apert_pix = (
            2. * np.degrees(np.arccos(1.-telescope.sol_angle_pix/2./math.pi)))
        # deg

    if area is not None:
        telescope.area = area

    if int_time is not None:
        telescope.int_time = int_time

    if isinstance(efficiency, pd.DataFrame):
        # Sorted to allow for interpolation
        efficiency.sort_index(axis=0, ascending=True, inplace=True)
        # The first column must be wavelength in nm
        wvl = np.array(efficiency.iloc[:, 0])
        telescope.wvl_cher = wvl
        telescope.wvl_ini = wvl[0]
        telescope.wvl_step = wvl[1] - wvl[0]
        telescope.wvl_fin = wvl[-1] + telescope.wvl_step/2.
        # -> wvl = np.arange(wvl_ini, wvl_fin, wvl_step)
        if not np.all(np.diff(wvl) == telescope.wvl_step):
            raise ValueError(
                "The wavelength discretization step must be constant.")

        # The second column must be efficiency in decimal fraction
        eff = np.array(efficiency.iloc[:, 1])
        if (not np.all(eff >= 0.)) or (not np.all(eff <= 1.)):
            raise ValueError(
                "Efficiency must be positive and in decimal fraction.")
        telescope.eff_cher = eff
        telescope.eff_fluo = np.interp(telescope.wvl_fluo, wvl, eff, left=0.,
                                       right=0.)

    else:
        ValueError("The input efficiency data is not valid.")

    return telescope


# Class #######################################################################
class _Telescope:
    """
    Telescope object containing the characteristics of a Cherenkov telescope.

    Class attributes
    ----------------
    tel_type : str
        Name of the subclass of Telescope. Presently only the parent
        class Telescope and the IACT and GridElement subclasses are available.
        More subclasses to be implemented.
    apert : float
        Angular diameter in degrees of the telescope field of view.
    area : float
        Detection area in m^2 (e.g., mirror area of an IACT).
    N_pix : int
        Number of camera pixels.
    int_time : float
        Integration time in microseconds of camera frames.
    sol_angle : float
        Telescope field of view in stereoradians.
    sol_angle_pix : float
        Pixel field of view in steresorians.
    apert_pix : float
        Angular diameter in degrees of the pixel FoV.
    wvl_ini : float
        Initial wavelength in nm of the detection efficiency data.
    wvl_fin : float
        Final wavelength in nm of the detection efficiency data.
    wvl_step : float
        Step size in nm of the detection efficiency data.
    wvl_fluo : ndarray
        Array containing the wavelengths of the 34 fluorescence bands
        included in the model.
    eff_fluo : ndarray
        Array containing the detection efficiency at these 34
        wavelenghts.
    wvl_cher : ndarray
        Array containing the range of wavelengths in nm defined by
        wvl_ini, wvl_fin and wvl_step
    eff_cher : ndarray
        Array containing the detection efficiency data in this range
        used to compute the Cherenkov signal.

    Object attributes
    -----------------
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
        x coordinate of a unit vector parallel to the telescope pointing
        direction.
    uy : float
        y coordinate of a unit vector parallel to the telescope pointing
        direction.
    uz : float
        z coordinate of a unit vector parallel to the telescope pointing
        direction.
    sin_theta = float
        Sine of theta (cosine of alt).
    cos_theta = float
        Cosine of theta (sine of alt).
    sin_az = float
        Sine of az.
    cos_az = float
        Cosine of az.
    phi_right : float
        Position angle phi in degrees of the right-hand direction from
        the telescope point of view.

    Methods
    -------
    copy : Copy the Telescope object, but with optional changes.
    hor_to_FoV : Convert cartesian coordinates from horizontal system to FoV
        system.
    FoV_to_hor : Convert cartesian coordinates from FoV system to horizontal
        system.
    thetaphi_to_altaz : Convert FoV coordinates theta/phi to horizontal
        coordinates alt/az.
    altaz_to_thetaphi : Convert horizontal coordinates alt/az to FoV
        coordinates theta/phi.
    spherical : Calculate the spherical coordinates in both horizontal and FoV
        systems.
    abs_to_rel : Calculate the x, y, z coordinates relative to the telescope
        position from the 'absolute' x, y, z coordinates.
    distance : Calculate the distance in km between the point x, y, z
        ('absolute' coordinates) and the telescope position.

    See also
    --------
    _Telescope : Telescope class.
    """
    # Default values of the class. They may be redefined in subclasses
    tel_type = None   # Generic telescope
    apert = 10.  # deg
    area = 100.  # m^2
    N_pix = 1500
    int_time = 0.01  # us

    sol_angle = 2. * math.pi*(1. - math.cos(math.radians(apert) / 2.))  # str
    sol_angle_pix = sol_angle / N_pix    # str
    apert_pix = 2. * np.degrees(np.arccos(1.-sol_angle_pix/2./math.pi))  # deg
    # Delta_pix = math.sqrt(sol_angle_pix / 2.)
    # Delta_r = math.sqrt(sol_angle / 2 / math.pi)

    wvl_ini = 290.  # nm
    wvl_fin = 430.  # nm
    wvl_step = 3.   # nm only used to integrate the Cherenkov light

    # 34 fluorescence bands included in the model
    wvl_fluo = np.array([296, 298, 302, 308, 312, 314, 316,
                         318, 327, 329, 331, 334, 337, 346,
                         350, 354, 358, 366, 367, 371, 376,
                         381, 386, 388, 389, 391, 394, 400,
                         405, 414, 420, 424, 427, 428])
    # 100% efficiency assumed at these 34 wavelengths
    eff_fluo = np.ones(34)

    wvl_cher = np.arange(wvl_ini, wvl_fin, wvl_step)  # 47 wavelengths
    # 100% efficiency assumed
    eff_cher = np.ones_like(wvl_cher)

    # Methods #################################################################
    def copy(self, **kwargs):
        """
        Copy a Telescope object, but with optional changes.

        Parameters
        ----------
        **kwargs {x, y, z, ...}
            Optional changes to the original telescope
            attributes, including class attributes.

        Returns
        -------
        Telescope object.
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

        telescope_c = Telescope(**kwargs)

        if 'efficiency' not in kwargs:
            telescope_c.wvl_ini = self.wvl_ini
            telescope_c.wvl_step = self.wvl_step
            telescope_c.wvl_fin = self.wvl_fin
            telescope_c.wvl_cher = self.wvl_cher
            telescope_c.eff_cher = self.eff_cher
            telescope_c.eff_fluo = self.eff_fluo

        return telescope_c

    def altaz_to_thetaphi(self, alt, az):
        """
        Convert polar horizontal coordinates alt, az to FoV coordinates
        theta, phi.

        Parameters
        ----------
        alt : float or array_like.
        az : float or array_like.

        Returns
        -------
        theta, phi : float or array_like.

        See also
        --------
        Telescope.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Telescope.thetaphi_to_altaz : Convert FoV coordinates theta, phi to
            horizontal coordinates alt, az.
        """
        alt = np.radians(alt)
        az = np.radians(az)

        x_hor = np.cos(alt) * np.sin(az)
        y_hor = np.cos(alt) * np.cos(az)
        z_hor = np.sin(alt)

        x_FoV, y_FoV, z_FoV = self.hor_to_FoV(x_hor, y_hor, z_hor)

        theta = self.zr_to_theta(z_FoV, 1.)
        phi = self.xy_to_phi(x_FoV, y_FoV) + self.phi_right - 360.
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
        x_hor : float or array_like.
        y_hor : float or array_like.
        z_hor : float or array_like.

        Returns
        -------
        x_FoV, y_FoV, z_FoV : float or array_like.

        See also
        --------
        Telescope.FoV_to_hor : Convert cartesian coordinates from FoV system to
            horizontal system.
        Telescope.altaz_to_thetaphi : Convert horizontal coordinates alt, az to
            FoV coordinates theta, phi.
        """
        sin_theta = self.sin_theta
        cos_theta = self.cos_theta
        sin_az = self.sin_az
        cos_az = self.cos_az

        x_FoV = cos_az * x_hor - sin_az * y_hor
        y_FoV = (cos_theta * sin_az * x_hor + cos_theta * cos_az * y_hor
                 - sin_theta * z_hor)
        z_FoV = (sin_theta * sin_az * x_hor + sin_theta * cos_az * y_hor
                 + cos_theta * z_hor)

        return x_FoV, y_FoV, z_FoV

    def thetaphi_to_altaz(self, theta, phi):
        """
        Convert FoV coordinates theta, phi to horizontal coordinates alt, az.

        Parameters
        ----------
        theta : float or array_like.
        phi : float or array_like.

        Returns
        -------
        alt, az : float or array_like.

        See also
        --------
        Telescope.FoV_to_hor : Convert cartesian coordinates from FoV system to
            horizontal system.
        Telescope.altaz_to_thetaphi : Convert horizontal coordinates alt, az to
            FoV coordinates theta, phi.
        """
        theta = np.radians(theta)
        phi = np.radians(phi - self.phi_right + 360.)

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
        x_FoV : float or array_like.
        y_FoV : float or array_like.
        z_FoV : float or array_like.

        Returns
        -------
        x_hor, y_hor, z_hor : float or array_like.

        See also
        --------
        Telescope.hor_to_FoV : Convert cartesian coordinates from horizontal
            system to FoV system.
        Telescope.thetaphi_to_altaz : Convert FoV coordinates theta, phi to
            horizontal coordinates alt, az.
        """
        sin_theta = self.sin_theta
        cos_theta = self.cos_theta
        sin_az = self.sin_az
        cos_az = self.cos_az

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
        """
        x_rel, y_rel, z_rel = self.abs_to_rel(x, y, z)
        return np.sqrt(x_rel**2 + y_rel**2 + z_rel**2)

    def spherical(self, x, y, z):
        """
        Calculate the spherical coordinates in both horizontal and FoV systems
        from the 'absolute' x, y, z coordinates.

        Parameters
        ----------
        x, y, z : float or array_like.

        Returns
        -------
        distance, alt, az, theta, phi : float or array_like.
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
        phi = self.xy_to_phi(x_FoV, y_FoV) + self.phi_right - 360.
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
        """
        from ._tools import zr_to_theta
        return zr_to_theta(z, r)

    def xy_to_phi(self, x, y):
        """
        Calculate the angle phi in degrees [-90, 270) of the xy projection of a
        vector, where phi is defined from the x axis towards the y axis
        (anticlockwise)
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
        track : Track object or Shower object.

        Returns
        -------
        Projection object.

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
        track : Track object or Shower object.
        axes : bool, default True
            Show the axes of both coordinate systems of reference.
        max_theta : float, default 30 degrees
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
        projection = sm.Projection(self, track)
        from ._tools import show_projection
        return projection, (show_projection(projection, None, False, axes,
                                            max_theta, X_mark))


# Subclasses ##################################################################
# Presently only the IACT and grid_elem subclasses are available.
# More subclasses to be implemented.
class _IACT(_Telescope):
    # Default values of IACT
    tel_type = 'IACT'
    apert = 8.  # deg
    area = 113.097  # m^2
    N_pix = 1800
    # int_time = 0.01  # us

    sol_angle = 2. * math.pi*(1. - math.cos(math.radians(apert)/2.))  # str
    sol_angle_pix = sol_angle / N_pix  # str
    apert_pix = 2. * np.degrees(np.arccos(1.-sol_angle_pix/2./math.pi))  # deg
    Delta_pix = math.sqrt(sol_angle_pix / 2.)
    Delta_r = math.sqrt(sol_angle / 2 / math.pi)

    wvl_ini = 280.  # nm
    wvl_fin = 600.  # nm
    wvl_step = 3.   # nm only used to integrate the Cherenkov light

    # 34 fluorescence bands included in the model
    # wvl_fluo = np.array([296, 298, 302, 308, 312, 314, 316,
    #                      318, 327, 329, 331, 334, 337, 346,
    #                      350, 354, 358, 366, 367, 371, 376,
    #                      381, 386, 388, 389, 391, 394, 400,
    #                      405, 414, 420, 424, 427, 428])

    # Detection efficiency taken from CTA data at these 34 wavelengths
    eff_fluo = np.array([0.252489433, 0.263035139, 0.287398782, 0.308913092,
                         0.319016638, 0.323187842, 0.326638232, 0.329752346,
                         0.339552709, 0.341029170, 0.342454324, 0.344086960,
                         0.345252040, 0.347248391, 0.347256505, 0.346686550,
                         0.346354164, 0.349208764, 0.349801116, 0.352296156,
                         0.354879676, 0.356753586, 0.353959270, 0.352900969,
                         0.352432398, 0.351494414, 0.350073889, 0.346680335,
                         0.342513534, 0.335703253, 0.330467935, 0.326751911,
                         0.323959630, 0.323028670])

    # 107 wavelengths, ending in 598nm
    wvl_cher = np.arange(wvl_ini, wvl_fin, wvl_step)
    # Detection efficiency taken from CTA data at these 107 wavelengths
    eff_cher = np.array([0.015491090, 0.079299180, 0.129636709, 0.172271597,
                         0.207375577, 0.236202184, 0.263035139, 0.280541802,
                         0.294669433, 0.305342911, 0.314872346, 0.321098876,
                         0.326638232, 0.331242468, 0.334854807, 0.337795491,
                         0.340290566, 0.342454324, 0.344086960, 0.345252040,
                         0.346172895, 0.346778694, 0.347248391, 0.347255182,
                         0.347044138, 0.346506611, 0.346354164, 0.346979403,
                         0.348025435, 0.349801116, 0.351659605, 0.353400273,
                         0.354879676, 0.356003591, 0.356271574, 0.354537850,
                         0.352900969, 0.351494414, 0.350073889, 0.348520911,
                         0.346680335, 0.344180639, 0.341759844, 0.339601025,
                         0.337372559, 0.334868408, 0.332311345, 0.329540303,
                         0.326751911, 0.323959630, 0.320980527, 0.317070704,
                         0.313088141, 0.308639570, 0.304189576, 0.298629946,
                         0.292593521, 0.286598981, 0.280644502, 0.275022087,
                         0.269524770, 0.264660241, 0.259794163, 0.255157961,
                         0.250526184, 0.246266668, 0.242256909, 0.238286004,
                         0.234340729, 0.230033882, 0.225450238, 0.219591053,
                         0.212757397, 0.205538060, 0.196889181, 0.185468854,
                         0.173996418, 0.164205860, 0.155335908, 0.147509786,
                         0.138970850, 0.132442597, 0.126951003, 0.121893936,
                         0.117054131, 0.113109258, 0.109682504, 0.106428167,
                         0.103273833, 0.100303912, 0.097453247, 0.094448853,
                         0.091341851, 0.088381484, 0.085533194, 0.082668619,
                         0.079789755, 0.077237853, 0.074972054, 0.072399174,
                         0.069519165, 0.066662492, 0.063829159, 0.061106831,
                         0.058511426, 0.057261531, 0.057550245])
    pass


class _GridElement(_Telescope):
    # Default values
    tel_type = 'GridElement'
    # theta = 0.  # deg  It is set to zero by default when Grid is called
    apert = 180.  # deg
    # area = 100. # m^2 It is set to one grid cell when Grid is called
    N_pix = 1
    int_time = 10.  # us

    sol_angle = 2. * math.pi    # str
    sol_angle_pix = sol_angle   # str
    apert_pix = apert           # deg
    Delta_pix = math.sqrt(math.pi)
    Delta_r = 1.

    # wvl_ini = 290.  # nm
    # wvl_fin = 430.  # nm
    # wvl_step = 3.   # nm only used to integrate the Cherenkov light

    # 34 fluorescence bands included in the model
    # wvl_fluo = np.array([296, 298, 302, 308, 312, 314, 316,
    #                      318, 327, 329, 331, 334, 337, 346,
    #                      350, 354, 358, 366, 367, 371, 376,
    #                      381, 386, 388, 389, 391, 394, 400,
    #                      405, 414, 420, 424, 427, 428])

    # 100% efficiency assumed at these 34 wavelengths
    # eff_fluo = np.ones(34)

    # wvl_cher = np.arange(wvl_ini, wvl_fin, wvl_step) # 47 wavelengths
    # 100% efficiency assumed
    # eff_cher = np.ones_like(wvl_cher)

    pass

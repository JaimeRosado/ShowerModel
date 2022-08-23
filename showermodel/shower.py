# coding: utf-8

import showermodel as sm
import showermodel.constants as ct
import matplotlib.pyplot as plt
from .event import _show_distribution


# Default values for Shower
_Shower__theta = ct.config['Shower']['theta']
_Shower__alt = ct.config['Shower'].get('alt') # optional parameter
_Shower__az = ct.config['Shower']['az']
_Shower__x0 = ct.config['Shower']['x0']
_Shower__y0 = ct.config['Shower']['y0']
_Shower__xi = ct.config['Shower']['xi']
_Shower__yi = ct.config['Shower']['yi']
_Shower__zi = ct.config['Shower'].get('zi') # optional parameter
_Shower__E = ct.config['Shower']['E']
_Shower__prf_model = ct.config['Shower']['prf_model']
_Shower__X_max = ct.config['Shower'].get('X_max')  # optional parameter
_Shower__X0_GH = ct.config['Shower'].get('X0_GH')  # optional parameter
_Shower__lambda_GH = ct.config['Shower'].get('lambda_GH')  # optional parameter
_Shower__h0 = ct.config['Atmosphere']['h0']
_Shower__h_top = ct.config['Atmosphere'].get('h_top') # optional parameter
_Shower__N_steps = ct.config['Atmosphere']['N_steps']
_Shower__atm_model = ct.config['Atmosphere']['atm_model']
_Shower__rho_w_sl = ct.config['Atmosphere']['rho_w_sl']
_Shower__h_scale = ct.config['Atmosphere']['h_scale']

# Default values for Signal and Event
_Signal__atm_trans = ct.config['Signal']['atm_trans']
_Signal__tel_eff = ct.config['Signal']['tel_eff']
_Signal__wvl_ini = ct.config['Signal']['wvl_ini']
_Signal__wvl_fin = ct.config['Signal']['wvl_fin']
_Signal__wvl_step = ct.config['Signal']['wvl_step']

# Default values for Grid
_Grid__obs_name = ct.config['Grid'].get('obs_name') # optional parameter
_Grid__tel_type = ct.config['Grid']['tel_type']
_Grid__x_c = ct.config['Grid']['x_c']
_Grid__y_c = ct.config['Grid']['y_c']
_Grid__z_c = ct.config['Grid']['z_c']
_Grid__size_x = ct.config['Grid']['size_x']
_Grid__size_y = ct.config['Grid']['size_y']
_Grid__N_x = ct.config['Grid']['N_x']
_Grid__N_y = ct.config['Grid']['N_y']
_Grid__theta = ct.config['Telescope']['theta']
_Grid__alt = ct.config['Telescope'].get('alt') # optional parameter
_Grid__az = ct.config['Telescope']['az']

# Class #######################################################################
class Shower:
    """
    Object containing a discretization of a shower.

    It includes the atmosphere, both the track and profile of the shower
    as well as its fluorescence and Cherenkov light production.

    Use sm.Shower() to construct the default Shower object.

    Parameters
    ----------
    E : float, default 10000000
        Energy of the primary particle in MeV.
    theta : float, default 0
        Zenith angle in degrees of the apparent position of the source.
    alt : float, default None
        Altitude in degrees of the apparent position of the source. If None,
        theta is used. If given, theta is overwritten.
    az : float, default 0
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0 : float, default 0
        East coordinate in km of shower impact point at ground.
    y0 : float, default 0
        North coordinate in km of shower impact point at ground.
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
    prf_model : 'Greisen', 'Gaisser-Hillas' or DataFrame, default 'Greisen'
        Profile model to be used. If 'Greisen', the Greisen function
        for electromagnetic showers is used. If 'Gaisser-Hillas', the
        Gaisser-Hillas function for hadron-induced showers is used.
        If a DataFrame is given, it should have two columns, the first
        one with the slant depth in g/cm2 and the second one with dE/dX
        in MeV.cm2/g..
    X_max : float, default None
        Slant depth in g/cm^2 at shower maximum. If None and prf_model
        is 'Greisen' or 'Gaisser-Hillas', a typical value of X_max for
        gamma or proton showers is calculated from the radiation length
        lambda_r = 36.7 g/cm^2 and the critical energy E_c = 81 MeV.
    X0_GH : float, default None
        X0 parameter in g/cm2 to be used when prf_model=='Gaisser-
        Hillas'. If None, a typical value for the input energy is used.
    lambda_GH : float, default None
        Lambda parameter in g/cm2 to be used when prf_model=='Gaisser-
        Hillas'. If None, a typical value for the input energy is used.
    atmosphere : Atmosphere, default None
        Atmosphere object to be used. If None, a new Atmosphere object
        is generated. If given, h0, h_top, N_steps, atm_model, rho_w_sl
        and h_scale are ignored.
    h0 : float, default 0.0
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
    atmosphere : Atmosphere
        Atmosphere object that is used.
    h0 : float
        Ground level in km above sea level.
    h_top : float
        Top level of the atmosphere in km above sea level.
    N_steps : int
        Number of discretization steps.
    h_step : float
        Size of discretization step in km.
    atm_model : int or DataFrame
        Atmospheric model.
    track : Track
        Track object that is generated.
    profile : Profile
        Profile object that is generated.
    E : float
        Energy of the primary particle in MeV.
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0, y0, z0 : float or None
        Coordinates in km of shower impact point at ground (z0=0).
        Set to None for ascending showers beginning at zi>0.
    xi, yi, zi : float
        Coordinates in km of the first interaction point of the shower.
    prf_model : 'Greisen', 'Gaisser-Hillas' or DataFrame.
    X_max : float
        Slant depth in g/cm^2 at shower maximum.
    X0_GH : float
        X0 parameter in g/cm2 for prf_model=='Gaisser-Hillas'.
    lambda_GH : float
        Lambda parameter in g/cm2 for prf_model=='Gaisser-Hillas'.
    fluorescence : Fluorescence
        Fluorescence object that is generated.
    cherenkov : Cherenkov
        Cherenkov object that is generated.

    Methods
    -------
    copy()
        Copy a Shower object, but with optional changes.
    Projection()
        Make a Projection object containing the coordinates of a
        shower track relative to a telescope position.
    Signal()
        Make a Signal object containing the signal produced by the shower
        detected by a telescope.
    Event()
        Make an Event object containing the characteristics of the shower,
        an observatory and the signal produced by the shower in each telescope.
    show_profile()
        Show the shower profile, both number of charged particles
        and energy deposit, as a function of slant depth.
    show_light_production()
        Show the production of both Cherenkov and fluorescence photons
        as a function of slant depth.
    show_projection()
        Make a Projection object and show it.
    show_signal()
        Make a Signal object and show it.
    show_geometry2D()
        Show a 2D plot of the shower track and input telescope positions.
    show_geometry3D()
        Show a 3D plot of the shower track and input telescopes positions.
    show_distribution()
        Make a GridEvent object and show the distribution of photons
        per m^2 in a 1D or 2D plot.

    See also
    --------
    Atmosphere : DataFrame containing the atmosphere discretization.
    Track : DataFrame containing a shower track discretization.
    Profile : DataFrame containing a shower profile discretization.
    """
    def __init__(self, E=__E, theta=__theta, alt=__alt, az=__az, x0=__x0,
                 y0=__y0, xi=__x0, yi=__y0, zi=__zi, prf_model=__prf_model,
                 X_max=__X_max, X0_GH=__X0_GH, lambda_GH=__lambda_GH,
                 atmosphere=None, h0=__h0, h_top=__h_top, N_steps=__N_steps,
                 atm_model=__atm_model, rho_w_sl=__rho_w_sl,
                 h_scale=__h_scale):
        _shower(self, E, theta, alt, az, x0, y0, xi, yi, zi, prf_model, X_max,
                X0_GH, lambda_GH, atmosphere, h0, h_top, N_steps, atm_model,
                rho_w_sl, h_scale)

    def copy(self, **kwargs):
        """
        Copy a Shower object, but with optional changes.

        Parameters
        ----------
        **kwargs
            Optional key arguments to be passed to the constructors of
            the different attributes of the Shower object.

        Returns
        -------
        shower : Shower

        See also
        --------
        Shower : Make a discretization of a shower.
        """
        return _copy(self, **kwargs)

    def Projection(self, telescope):
        """
        Obtain the coordinates of a shower track relative to a telescope
        position in both zenith and camera projection and determine the
        fraction of the track within the telescope field of view.

        Parameters
        ----------
        telescope : Telescope, mandatory
            Telescope object to be used.

        Returns
        -------
        projection : Projection
        (ax1, ax2) : AxesSubplot

        See also
        --------
        Projection.show
        """
        return sm.Projection(telescope, self.track)

    def Signal(self, telescope, atm_trans=_Signal__atm_trans,
               tel_eff=_Signal__tel_eff, wvl_ini=_Signal__wvl_ini,
               wvl_fin=_Signal__wvl_fin, wvl_step=_Signal__wvl_step):
        """
        Calculate the signal produced by the shower detected by a telescope.

        Parameters
        ----------
        telescope : Telescope, mandatory
            Telescope object to be used.
        atm_trans : bool, default True
            Include the atmospheric transmission.
        tel_eff : bool, default True
            Include the telescope efficiency. If False, 100% efficiency is
            assumed for a given wavelength interval.
        wvl_ini : float, default 290
            Initial wavelength in nm of the interval to calculate the signal
            when tel_eff==False.
        wvl_fin : float, default 430
            Final wavelength in nm of the interval to calculate the signal when
            tel_eff==False.
        wvl_step : float, default 3
            Discretization step in nm of the interval to calculate the signal
            when tel_eff==False.

        Returns
        -------
        signal : Signal
        """
        return sm.Signal(telescope, self, atm_trans, tel_eff,
                         wvl_ini, wvl_fin, wvl_step)

    def Event(self, observatory, atm_trans=_Signal__atm_trans,
              tel_eff=_Signal__tel_eff, wvl_ini=_Signal__wvl_ini,
              wvl_fin=_Signal__wvl_fin, wvl_step=_Signal__wvl_step):
        """
        Make an Event object containing the characteristics of a shower, an
        observatory and the signal produced by the shower in each telescope.

        Parameters
        ----------
        observatory : Observatory, mandatory
            Observatory object (may be a Grid object).
        atm_trans : bool, default True
            Include the atmospheric transmission to calculate the signals.
        tel_eff : bool, default True
            Include the telescope efficiency to calculate the signals.
            If False, 100% efficiency is assumed for a given
            wavelength interval.
<<<<<<< Updated upstream
        **kwargs : {wvl_ini, wvl_fin, wvl_step}
            These parameters will modify the wavelength interval when
            tel_eff==False. If None, the wavelength interval defined in the
            telescope is used.
=======
        wvl_ini : float, default 290
            Initial wavelength in nm of the interval to calculate the signal
            when tel_eff==False.
        wvl_fin : float, default 430
            Final wavelength in nm of the interval to calculate the signal when
            tel_eff==False.
        wvl_step : float, default 3
            Discretization step in nm of the interval to calculate the signal
            when tel_eff==False.
>>>>>>> Stashed changes

        Returns
        -------
        event : Event
        """
        return sm.Event(observatory, self, atm_trans, tel_eff,
                        wvl_ini, wvl_fin, wvl_step)

    def show_projection(self, telescope, shower_Edep=True, axes=True,
                        max_theta=30., X_mark='X_max'):
        """
        Make a Projection object and show it.

        Parameters
        ----------
        telescope : Telescope, mandatory
            Telescope object to be used.
        shower_Edep : bool, default True
            Make the radii of the shower track points proportional to the
            energy deposited in each step length.
        axes : bool, default True
            Show the axes of both frames of reference.
        max_theta : float, default 30 degrees
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
<<<<<<< Updated upstream
            made in the figure, default X_max. If X_mark=None, no mark
            is included.
=======
            marked in the figure. If set to None, no mark is included.
            By default, the mark is placed at X_max.
>>>>>>> Stashed changes

        Returns
        -------
        projection : Projection
        (ax1, ax2) : PolarAxesSubpot

        See also
        --------
        Projection.show
        """
        if X_mark == 'X_max':
            X_mark = self.X_max
        projection = sm.Projection(telescope, self.track)
        profile = self.profile
        from ._tools import show_projection
        return projection, (show_projection(projection, profile, shower_Edep,
                                            axes, max_theta, X_mark))

    def show_profile(self):
        """
        Show the shower profile, both number of charged particles and energy
        deposit, as a function of slant depth.

        Returns
        -------
        (ax1, ax2) : AxesSubplot
        """
        return self.profile.show()

    def show_light_production(self):
        """
        Show the production of both Cherenkov and fluorescence photons in the
        290 - 430 nm range as a function of slant depth.

        Returns
        -------
        (ax1, ax2) : AxesSubplot
        """
        # Cherenkov
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax1.plot(self.profile.X, self.cherenkov.N_ph, 'r-')
        ax1.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax1.axes.yaxis.set_label_text(
            "Cherenkov production (photons·cm$^2$/g)")

        # Fluorescence
        ax2.plot(self.profile.X, self.fluorescence.sum(axis=1), 'b-')
        ax2.axes.xaxis.set_label_text("Slant depth (g/cm$^2$)")
        ax2.axes.yaxis.set_label_text(
            "Fluorescence production (Photons·cm$^2$/g)")
        plt.tight_layout()
        return (ax1, ax2)

    def show_signal(self, telescope, atm_trans=_Signal__atm_trans,
                    tel_eff=_Signal__tel_eff, wvl_ini=_Signal__wvl_ini,
                    wvl_fin=_Signal__wvl_fin, wvl_step=_Signal__wvl_step):
        """
        Make a Signal object and show it.

        Parameters
        ----------
        telescope : Telescope
            Telescope object to be used.
        atm_trans : bool, default True
            Include the atmospheric transmission.
        tel_eff : bool, default True
            Include the telescope efficiency. If False, 100% efficiency is
            assumed for a given wavelength interval.
<<<<<<< Updated upstream
        **kwargs : {wvl_ini, wvl_fin, wvl_step}
            These parameters will modify the wavelength interval when
            tel_eff==False. If None, the wavelength interval defined in the
            telescope is used.
=======
        wvl_ini : float, default 290
            Initial wavelength in nm of the interval to calculate the signal
            when tel_eff==False.
        wvl_fin : float, default 430
            Final wavelength in nm of the interval to calculate the signal
            when tel_eff==False.
        wvl_step : float, default 3
            Discretization step in nm of the interval to calculate the signal
            when tel_eff==False.
>>>>>>> Stashed changes

        Returns
        -------
        signal : Signal
        (ax1, ax2) : AxesSubplot
        """
        signal = sm.Signal(telescope, self, atm_trans, tel_eff,
                           wvl_ini, wvl_fin, wvl_step)
        ax1, ax2 = signal.show()
        return signal, (ax1, ax2)

    def show_geometry2D(self, observatory, x_min=-1., x_max=1., y_min=-1,
                        y_max=1., X_mark='X_max', shower_Edep=True,
                        tel_index=False):
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
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure. By default, the mark is placed at X_max.
        shower_Edep : bool, default True
            Make the radii of the shower track points proportional to the
            energy deposited in each step length.
        tel_index : bool, default True
            Show the telescope indexes together the telescope position points.

        Returns
        -------
        ax : AxesSubplot
        """
        if X_mark == 'X_max':
            X_mark = self.X_max
        from ._tools import show_geometry
        return show_geometry(self, observatory, '2d', x_min, x_max, y_min,
                             y_max, X_mark, shower_Edep, False, tel_index,
                             False, False)

    def show_geometry3D(self, observatory, x_min=-1., x_max=1., y_min=-1,
                        y_max=1., X_mark='X_max', shower_Edep=True,
                        xy_proj=True, pointing=False):
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
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure. By default, the mark is placed at X_max.
        shower_Edep : bool, default True
            Make the radii of the shower track points proportional to the
            energy deposited in each step length.
        xy_proj : bool, default True
            Show the xy projection of the shower track.
        pointing : bool, default False
            Show the telescope axes.

        Returns
        -------
        ax : Axes3DSubplot
        """
        if X_mark == 'X_max':
            X_mark = self.X_max
        from ._tools import show_geometry
        return show_geometry(self, observatory, '3d', x_min, x_max, y_min,
                             y_max, X_mark, shower_Edep, False, False, xy_proj,
                             pointing)

    def show_distribution(self, grid=None, telescope=None,
                          tel_type=_Grid__tel_type,
                          x_c=_Grid__x_c, y_c=_Grid__y_c, z_c=_Grid__z_c,
                          size_x=_Grid__size_x, size_y=_Grid__size_y,
                          N_x=_Grid__N_x, N_y=_Grid__N_y, theta=_Grid__theta,
                          alt=_Grid__alt, az=_Grid__az,
                          atm_trans=_Signal__atm_trans, tel_eff=False,
                          wvl_ini=_Signal__wvl_ini, wvl_fin=_Signal__wvl_fin,
                          wvl_step=_Signal__wvl_step):
        """
        Make an Event from a Grid object and show the distribution of photons
        (or photoelectrons) per m^2 in an either 1D or 2D plot, depending on
        the grid dimensions.

        Parameters
        ----------
<<<<<<< Updated upstream
        grid : Grid
            If None, a new Grid object is generated from the specified
            dimensions and telescope characteristics.
            If given, {telescope, tel_type, ..., N_x, N_y} are not used.
        telescope : Telescope
            When grid==None. If telescope==None, the Grid object is constructed
            based on the default GridElement object.
        x_c : float
            x coordinate in km of the center of the grid.
        y_c : float
            y coordinate in km of the center of the grid.
        z_c : float
=======
        grid : Grid or None, default None
            Grid object to generate the distribution. If None, a new Grid
            object is generated from the specified dimensions and the
            characteristics of the telescope. If given, telescope, size_x,
            size_y, N_x, N_y are not used.
        telescope : Telescope, default None
            Telescope object to be used to construct the grid. If None, the
            given tel_type telescope is used.
        tel_type : str, default 'GridElement'
            Type of telescope to be used when telescope==None.
        x_c : float, default 0
            East coordinate in km of the center of the grid.
        y_c : float, default 0
            North coordinate in km of the center of the grid.
        z_c : float, default 0
>>>>>>> Stashed changes
            Height of the grid in km above ground level.
        size_x : float, defaut 2
            Size of the grid in km across the x direction.
        size_y : float, default 2
            Size of the grid in km across the y direction.
        N_x : int, default 10
            Number of cells across the x direction.
        N_y : int, default 10
            Number of cells across the y direction.
        theta : float, default 0
            Zenith angle in degrees of the telescope pointing directions.
        alt : float, default None
            Altitude in degrees of the telescope pointing directions. If None,
            theta is used. If given, theta is overwritten.
        az : float, default 0
            Azimuth angle (from north, clockwise) in degrees of the telescope
            pointing directions.
        atm_trans : bool, default True
<<<<<<< Updated upstream
            Include the atmospheric transmission to transport photons.
        tel_eff : bool, default True
            Include the telescope efficiency to calculate the signal. If False,
            100% efficiency is assumed for a given wavelength interval.
        **kwargs : {wvl_ini, wvl_fin, wvl_step}
            These parameters will modify the wavelength interval when
            tel_eff==False. If None, the wavelength interval defined in the
            telescope is used.
=======
            Include the atmospheric transmision to transport photons.
        tel_eff : bool, default False
            Include the telescope efficiency to calculate the signal. If False,
            100% efficiency is assumed for a given wavelenght interval.
        wvl_ini : float, default 290
            Initial wavelength in nm of the interval to calculate the signal
            when tel_eff==False.
        wvl_fin : float, default 430
            Final wavelength in nm of the interval to calculate the signal when
            tel_eff==False.
        wvl_step : float, default 3
            Discretization step in nm of the interval to calculate the signal
            when tel_eff==False.
>>>>>>> Stashed changes

        Returns
        -------
        grid_event : Event
        ax : AxesSubplot
            If 1D grid.
        (ax1, ax2, cbar) : AxesSubplot and Colorbar
            If 2D grid.
        """
        if not isinstance(grid, sm.Grid):
            if grid is None:
                obs_name = _Grid__obs_name
                grid = sm.Grid(obs_name, telescope, tel_type, x_c, y_c, z_c,
                               size_x, size_y, N_x, N_y, theta, alt, az)
            else:
                raise ValueError('The input grid is not valid')

        grid_event = sm.Event(grid, self, atm_trans, tel_eff, wvl_ini, wvl_fin,
                              wvl_step)
        return grid_event, _show_distribution(grid_event)


# Auxiliary functions #########################################################
def _copy(shower, **kwargs):
    """
    Copy a Shower object, but with optional changes.
    """
    kwargs['E'] = kwargs.get('E', shower.E)
    # If 'alt' in kwargs and != None, theta is not used
    kwargs['theta'] = kwargs.get('theta', shower.theta)
    kwargs['az'] = kwargs.get('az', shower.az)

    # xi, yi, zi are used unless zi is not given but x0 or y0 are specified
    if kwargs.get('zi') is None and (kwargs.get('x0') is not None
                                     or kwargs.get('y0') is not None):
        kwargs['x0'] = kwargs.get('x0', shower.x0)
        kwargs['y0'] = kwargs.get('y0', shower.y0)
    else:
        kwargs['xi'] = kwargs.get('xi', shower.xi)
        kwargs['yi'] = kwargs.get('yi', shower.yi)
        kwargs['zi'] = kwargs.get('zi', shower.zi)
    kwargs['X_max'] = kwargs.get('X_max', shower.X_max)
    kwargs['prf_model'] = kwargs.get('prf_model', shower.prf_model)
    kwargs['X0_GH'] = kwargs.get('X0_GH', shower.X0_GH)
    kwargs['lambda_GH'] = kwargs.get('lambda_GH', shower.lambda_GH)

    # If no atmospheric parameter is given, then use that of the
    # original shower
    if (kwargs.get('atmosphere') is None and
        kwargs.get('h0') is None and
        kwargs.get('h_top') is None and
        kwargs.get('N_steps') is None and
        kwargs.get('atm_model') is None and
        kwargs.get('rho_w_sl') is None and
        kwargs.get('h_scale') is None):

        kwargs['atmosphere'] = shower.atmosphere

    # Otherwise, obtain the atsmosphere as usual,
    # whether atmosphere is given or not
    else:
        kwargs['h0'] = kwargs.get('h0', shower.h0)
        kwargs['h_top'] = kwargs.get('h_top', shower.h_top)
        kwargs['N_steps'] = kwargs.get('N_steps', shower.N_steps)
        kwargs['atm_model'] = kwargs.get('atm_model', shower.atm_model)
        kwargs['rho_w_sl'] = kwargs.get('rho_w_sl', shower.atmosphere.rho_w_sl)
        kwargs['h_scale'] = kwargs.get('h_scale', shower.atmosphere.h_scale)

    return Shower(**kwargs)


# Constructor #################################################################
def _shower(shower, E, theta, alt, az, x0, y0, xi, yi, zi, prf_model, X_max,
            X0_GH, lambda_GH, atmosphere, h0, h_top, N_steps, atm_model,
            rho_w_sl, h_scale):
    """
    Constructor of Shower object.

    It includes the shower profile and its fluorescence and Cherenkov light
    production.

    Parameters
    ----------
    shower : Shower
        Shower object.
    E : float
        Energy of the primary particle in MeV.
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0, y0, z0 : float or None
        Coordinates in km of shower impact point at ground (z0=0).
        Set to None for ascending showers beginning at zi>0.
    xi, yi, zi : float
        Coordinates in km of the first interaction point of the shower.
    prf_model : {'Greisen', 'Gaisser-Hillas'} or DataFrame
        If 'Greisen', the Greisen function for electromagnetic showers is used.
        If 'Gaisser-Hillas', the Gaisser-Hillas function for hadron-induced
        showers is used. If a DataFrame with an energy deposit profile is
        input, it must have two columns with the slant depth in g/cm2 and dE/dX
        in MeV.cm2/g. 
    X_max : float
        Slant depth in g/cm^2 at shower maximum. If None and prf_model is
        'Greisen' or 'Gaisser-Hillas', a typical value of X_max for gamma or
        proton showers is used. If None and a numerical energy deposit profile
        is input, lambda_r = 36.7 g/cm^2 is the radiation length E_c = 81 MeV
        is the critical energy.
    X0_GH : float
        X0 parameter in g/cm2 to be used when prf_model=='Gaisser-Hillas'.
        If None, a typical value for the input energy is used.
    lambda_GH : float
        Lambda parameter in g/cm2 to be used when prf_model=='Gaisser-Hillas'.
        If None, a typical value for the input energy is used.
    atmosphere : Atmosphere
        If None, a new Atmosphere object is generated.
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

    # shower = Shower()
    shower.E = E
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    shower.theta = theta
    shower.alt = alt
    shower.az = az

    # Atmosphere
    shower.atmosphere = atmosphere

    shower.h0 = atmosphere.h0
    shower.h_top = atmosphere.h_top
    shower.N_steps = atmosphere.N_steps
    shower.atm_model = atmosphere.atm_model

    # Shower track
    shower.track = sm.Track(theta, None, az, x0, y0, xi, yi, zi, atmosphere)
    shower.x0 = shower.track.x0
    shower.y0 = shower.track.y0
    shower.z0 = shower.track.z0
    shower.xi = shower.track.xi
    shower.yi = shower.track.yi
    shower.zi = shower.track.zi

    # Shower profile
    profile = sm.Profile(E, theta, None, prf_model, X_max, X0_GH, lambda_GH,
                         zi, atmosphere)
    shower.profile = profile

    shower.prf_model = profile.prf_model
    shower.X_max = profile.X_max
    shower.X0_GH = profile.X0_GH
    shower.lambda_GH = profile.lambda_GH

    # Fluorescence emission
    shower.fluorescence = shower.profile.Fluorescence()

    # Cherenkov emission
    shower.cherenkov = shower.profile.Cherenkov()

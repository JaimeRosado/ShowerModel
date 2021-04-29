# coding: utf-8

import showermodel as sm
import matplotlib.pyplot as plt


# Default values for shower
from .track import _theta, _az, _x0, _y0
from .profile import _E, _prf_model


# Constructor #################################################################
def Shower(E=_E, theta=_theta, alt=None, az=_az, x0=_x0, y0=_y0,
           prf_model=_prf_model, X_max=None, N_ch_max=None, X0_GH=None,
           lambda_GH=None, atmosphere=None, **kwargs):
    """
    Make a discretization of shower.

    It includes the shower profile and its fluorescence and Cherenkov light
    production.

    Parameters
    ----------
    E : float
        Energy of the primary particle.
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0 : float
        East coordinate in km of shower impact point at ground.
    y0 : float
        West coordinate in km of shower impact point at ground.
    prf_model : {'Greisen', 'Gaisser-Hillas'} or DataFrame
        If 'Greisen', the Greisen function for electromagnetic showers is used.
        If 'Gaisser-Hillas', the Gaisser-Hillas function for hadron-induced
        showers is used. If a DataFrame with an energy deposit profile is input,
        it must have two columns with the slant depth in g/cm2 and dE/dX in
        MeV.cm2/g. 
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
    atmosphere : Atmosphere object.
        If None, a new Atmosphere object is generated.
    **kwargs {h0, h_top, N_steps, model}
        Options to construct the new Atmosphere object when atmosphere==None.
        If None, the default Atmosphere object is used.

    Returns
    -------
    shower : Shower object.

    See also
    --------
    _Shower : Shower class.
    """
    from .atmosphere import _Atmosphere
    if isinstance(atmosphere, _Atmosphere):
        pass
    elif atmosphere is None:
        atmosphere = sm.Atmosphere(**kwargs)
    else:
        raise ValueError('The input atmosphere is not valid.')

    shower = _Shower()
    shower.E = E
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    shower.theta = theta
    shower.alt = alt
    shower.az = az
    shower.x0 = x0
    shower.y0 = y0

    # Atmosphere
    shower.atmosphere = atmosphere

    shower.h0 = atmosphere.h0
    shower.h_top = atmosphere.h_top
    shower.N_steps = atmosphere.N_steps
    shower.model = atmosphere.model

    # Shower track
    shower.track = sm.Track(theta, None, az, x0, y0, atmosphere)

    # Shower profile
    profile = sm.Profile(E, theta, None, prf_model, X_max, X0_GH, lambda_GH,
                         atmosphere)
    shower.profile = profile

    shower.prf_model = profile.prf_model
    shower.X_max = profile.X_max
    shower.X0_GH = profile.X0_GH
    shower.lambda_GH = profile.lambda_GH

    # Fluorescence emission
    shower.fluorescence = shower.profile.Fluorescence()

    # Cherenkov emission
    shower.cherenkov = shower.profile.Cherenkov()

    return shower


# Class #######################################################################
class _Shower:
    """
    Object containing a discretization of the atmosphere, both the track and
    profile of a shower as well as its fluorescence and Cherenkov light
    production.

    Use Shower to construct a Shower object.

    Attributes
    ----------
    atmosphere : Atmosphere object.
    h0 : float
        Ground level in km above sea level.
    h_top : float
        Top level of the atmosphere in km above sea level.
    N_steps : int
        Number of discretization steps.
    h_step : float
        Size of discretization step in km.
    model : int
        CORSIKA atmospheric model. Presently either 1 or 17. More models to
        be implemented.
    track : Track object.
    profile : Profile object.
    E : float
        Energy of the primary particle.
    theta : float
        Zenith angle in degrees of the apparent position of the source.
    alt : float
        Altitude in degrees of the apparent position of the source.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the apparent
        position of the source.
    x0 : float
        East coordinate in km of shower impact point at ground.
    y0 : float
        North coordinate in km of shower impact point at ground.
    prf_model : {'Greisen', 'Gaisser-Hillas'} or DataFrame.
    X_max : float
        Slant depth in g/cm^2 at shower maximum.
    X0_GH : float
        X0 parameter in g/cm2 for prf_model=='Gaisser-Hillas'.
    lambda_GH : float
        Lambda parameter in g/cm2 for prf_model=='Gaisser-Hillas'.
    fluorescence : Fluorescence object.
    cherenkov : Cherenkov object.

    Methods
    -------
    copy : Copy a Shower object, but with optional changes.
    Projection : Make a Projection object containing the coordinates of a
        shower track relative to a telescope position.
    Signal : Make a Signal object containing the signal produced by the shower
        detected by a telescope.
    Event : Make an Event object containing the characteristics of the shower,
        an observatory and the signal produced by the shower in each telescope.
    show_profile : Show the shower profile, both number of charged particles
        and energy deposit, as a function of slant depth.
    show_light_production : Show the production of both Cherenkov and
        fluorescence photons in the 290 - 430 nm range as a function of slant
        depth.
    show_projection : Make a Projection object and show it.
    show_signal : Make a Signal object and show it.
    show_event : Make an Event object and show it.
    show_geometry2D : Show a 2D plot of the shower track along with input
        telescope positions.
    show_geometry3D : Show a 3D plot of the shower track along with input
        telescopes positions.

    See also
    --------
    Atmosphere : Constructor of Atmosphere object.
    Track : Constructor of Track object.
    Profile : Constructor of Profile object.
    """

    def copy(self, **kwargs):
        """
        Copy a Shower object, but with optional changes.

        Depending on the input arguments, some attributes (or all them) will be
        aliases of those of the original Shower object.

        Parameters
        ----------
        **kwargs
            Optional key arguments to be passed to the constructors of
            the different attributes of the Shower object.

        Returns
        -------
        Shower object.

        See also
        --------
        Shower : Constructor of Shower object.
        """
        return _copy(self, **kwargs)

    def Projection(self, telescope):
        """
        Obtain the coordinates of a shower track relative to a telescope
        position in both zenith and camera projection and determine the
        fraction of the track within the telescope field of view.

        Parameters
        ----------
        telescope : Telescope object.

        Returns
        -------
        Projection object.
        (ax1, ax2) : AxesSubplot objects.

        See also
        --------
        Projection.show
        """
        return sm.Projection(telescope, self.track)

    def Signal(self, telescope, atm_trans=True, tel_eff=True, **kwargs):
        """
        Calculate the signal produced by the shower detected by a telescope.

        Parameters
        ----------
        telescope : Telescope object.
        atm_trans : bool, default True
            Include the atmospheric transmission.
        tel_eff : bool, default True
            Include the telescope efficiency. If False, 100% efficiency is
            assumed for a given wavelength interval.
        **kwargs {wvl_ini, wvl_fin, wvl_step}
            These parameters will modify the wavelenght interval when
            tel_eff==False. If None, the wavelength interval defined in the
            telescope is used.

        Results
        -------
        Signal object.
        """
        return sm.Signal(telescope, self, atm_trans, tel_eff, **kwargs)

    def Event(self, observatory, atm_trans=True, tel_eff=True, **kwargs):
        """
        Make an Event object containing the characteristics of a shower, an
        observatory and the signal produced by the shower in each telescope.

        Parameters
        ----------
        observatory : Observatory object (may be a Grid object).
        atm_trans : bool, default True
            Include the atmospheric transmision to calculate the signals.
        tel_eff : book, default True
            Include the telescope efficiency to calculate the signals.
            If False, 100% efficiency is assumed for a given
            wavelength interval.
        **kwargs {wvl_ini, wvl_fin, wvl_step}
            These parameters will modify the wavelenght interval when
            tel_eff==False. If None, the wavelength interval defined in the
            telescope is used.

        Results
        -------
        Event object.
        """
        return sm.Event(observatory, self, atm_trans, tel_eff, **kwargs)

    def show_projection(self, telescope, shower_size=True, axes=True,
                        max_theta=30., X_mark='X_max'):
        """
        Make a Projection object and show it.

        Parameters
        ----------
        telescope : Telescope object.
        shower_size : book, default True
            Make the radii of the shower track points proportional to the
            shower size.
        axes : book, default True
            Show the axes of both frames of reference.
        max_theta : float, default 30 degrees
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be ma
            ked in the figure, default X_max. If X_mark=None, no mark is
            included.

        Returns
        -------
        Projection object.
        (ax1, ax2) : PolarAxesSubpot objects.

        See also
        --------
        Projection.show
        """
        if X_mark == 'X_max':
            X_mark = self.X_max
        projection = sm.Projection(telescope, self.track)
        profile = self.profile
        from ._tools import show_projection
        return projection, (show_projection(projection, profile, shower_size,
                                            axes, max_theta, X_mark))

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

    def show_signal(self, telescope, atm_trans=True, tel_eff=True, **kwargs):
        """
        Make a Signal object and show it.

        Parameters
        ----------
        telescope : Telescope object.
        atm_trans : bool, default True
            Include the atmospheric transmision.
        tel_eff : bool, default True
            Include the telescope efficiency. If False, 100% efficiency is
            assumed for a given wavelength interval.
        **kwargs {wvl_ini, wvl_fin, wvl_step}
            These parameters will modify the wavelenght interval when
            tel_eff==False. If None, the wavelength interval defined in the
            telescope is used.

        Results
        -------
        Signal object.
        (ax1, ax2) : AxesSubplot objects.
        """
        signal = sm.Signal(telescope, self, atm_trans, tel_eff, **kwargs)
        ax1, ax2 = signal.show()
        return signal, (ax1, ax2)

    def show_geometry2D(self, observatory, x_min=-1., x_max=1., y_min=-1,
                        y_max=1., X_mark='X_max', shower_size=True,
                        tel_index=False):
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
            marked in the figure, default to X_max. If X_mark is set to None,
            no mark is included.
        shower_size : bool, default True
            Make the radii of the shower track points proportional to the
            shower size.
        tel_index : bool, default True
            Show the telescope indexes together the telescope position points.

        Returns
        -------
        AxesSubplot object.
        """
        if X_mark == 'X_max':
            X_mark = self.X_max
        from ._tools import show_geometry
        return show_geometry(self, observatory, '2d', x_min, x_max, y_min,
                             y_max, X_mark, shower_size, False, tel_index,
                             False, False)

    def show_geometry3D(self, observatory, x_min=-1., x_max=1., y_min=-1,
                        y_max=1., X_mark='X_max', shower_size=True,
                        xy_proj=True, pointing=False):
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
        shower_size : bool, default True
            Make the radii of the shower track points proportional to
            the shower size.
        xy_proj : bool, default True
            Show the xy projection of the shower track.
        pointing : bool, default False
            Show the telescope axes.

        Returns
        -------
        Axes3DSubplot object.
        """
        if X_mark == 'X_max':
            X_mark = self.X_max
        from ._tools import show_geometry
        return show_geometry(self, observatory, '3d', x_min, x_max, y_min,
                             y_max, X_mark, shower_size, False, False, xy_proj,
                             pointing)

    def show_distribution(self, grid=None, telescope=None,
                          tel_type='GridElement', x_c=0., y_c=0., z_c=0.,
                          theta=None, alt=None, az=None, size_x=2., size_y=2.,
                          N_x=10, N_y=10, atm_trans=True, tel_eff=False,
                          **kwargs):
        """
        Make a GridEvent object and show the distribution of photons
        (or photoelectrons) per m$^2$ in an either 1D or 2D plot, depending on
        the grid dimensions.

        Parameters
        ----------
        grid : Grid object.
            If None, a new Grid object is generated from the specificed
            dimensions and telescope characteristics.
            If given, {telescope, tel_type, ..., N_x, N_y} are not used.
        telescope : Telescope object (when grid==None)
            If None, the Grid object is constructed
            based on a Telescope object of type tel_type.
        tel_type : str
            Subclass of Telescope to be used when grid==None and
            telescope==None. Default to GridElement with 100% detection
            efficiency, FoV of 180 degrees around zenith and area of one grid
            cell. If tel_type==None, the parent class Telescope is used.
        x_c : float
            x coordinate in km of the center of the grid.
        y_c : float
            y coordinate in km of the center of the grid.
        z_c : float
            Height of the grid in km above ground level.
        theta : float
            Zenith angle in degrees of the telescope pointing directions.
        alt : float
            Altitude in degrees of the telescope pointing direction.
            If None, theta is used. If given, theta is overwritten.
        az : float
            Azimuth angle (from north, clockwise) in degrees of the telescope
            pointing direction.
        size_x : float
            Size of the grid in km across the x direction.
        size_y : float
            Size of the grid in km across the y direction.
        N_x : int
            Number of cells across the x direction.
        N_y : int
            Number of cells across the y direction.
        atm_trans : bool, default True
            Include the atmospheric transmision to transport photons.
        tel_eff : bool, default True
            Include the telescope efficiency to calculate the signal. If False,
            100% efficiency is assumed for a given wavelenght interval.
        **kwargs {wvl_ini, wvl_fin, wvl_step}
            These parameters will modify the wavelenght interval when
            tel_eff==False. If None, the wavelength interval defined in the
            telescope is used.

        Returns
        -------
        grid_event : GridEvent object.
        ax : AxesSubplot object (if 1D grid).
        (ax1, ax2, cbar) : AxesSubplot objects and Colorbar object
            (if 2D grid).
        """
        from .observatory import _Grid
        if not isinstance(grid, _Grid):
            if grid is None:
                grid = sm.Grid(telescope, tel_type, x_c, y_c, z_c, theta, alt,
                               az, size_x, size_y, N_x, N_y)
            else:
                raise ValueError('The input grid is not valid')

        grid_event = sm.Event(grid, self, atm_trans, tel_eff, **kwargs)
        return grid_event.show_distribution()


# Auxiliary functions #########################################################
def _copy(shower, atmosphere=None, **kwargs):
    """
    Copy a Shower object, but with optional changes.

    Depending on the input arguments, some attributes (or all them) will be
    aliases of those of the orginal Shower object.
    """
    E = kwargs.get('E', shower.E)
    if kwargs.get('alt') is None:
        theta = kwargs.get('theta', shower.theta)
        alt = 90. - theta
    else:
        alt = kwargs('alt')
        theta = 90. - alt
    az = kwargs.get('az', shower.az)
    x0 = kwargs.get('x0', shower.x0)
    y0 = kwargs.get('y0', shower.y0)
    X_max = kwargs.get('X_max', shower.X_max)
    prf_model = kwargs.get('prf_model', shower.prf_model)
    X0_GH = kwargs.get('X0_GH', shower.X0_GH)
    lambda_GH = kwargs.get('lambda_GH', shower.lambda_GH)
    h0 = kwargs.get('h0', shower.h0)
    h_top = kwargs.get('h_top', shower.h_top)
    N_steps = kwargs.get('N_steps', shower.N_steps)
    model = kwargs.get('model', shower.model)

    from .atmosphere import _Atmosphere
    if isinstance(atmosphere, _Atmosphere):
        # If the key argument atmosphere is passed with a valid atmosphere,
        # the function generates a new shower from scratch using that
        # atmosphere (h0, h_top, etc. not used)
        return Shower(E, theta, az, x0, y0, X_max, atmosphere)

    elif atmosphere is None:
        # If no key argument atmosphere is passed and some atmospheric
        # parameters are changed, a new atmosphere is generated and the shower
        # object is generated from scratch using those parameters
        if ((h0 != shower.h0) or (h_top != shower.h_top)
                or (N_steps != shower.N_steps) or (model != shower.model)):
            atmosphere = sm.Atmosphere(h0, h_top, N_steps, model)
            return Shower(E, theta, az, x0, y0, X_max, atmosphere)

        else:
            # If the atmospheric parameters are the same as the original
            # atmosphere a new Shower object is generated with an alias of the
            # original atmosphere
            shower_c = _Shower()
            shower_c.atmosphere = shower.atmosphere  # New alias

            # Attributes are overwritten
            shower_c.E = E
            shower_c.theta = theta
            shower_c.alt = alt
            shower_c.az = az
            shower_c.x0 = x0
            shower_c.y0 = y0
            shower_c.prf_model = prf_model
            shower_c.X_max = X_max
            shower_c.X0_GH = X0_GH
            shower_c.lambda_GH = lambda_GH
            shower_c.h0 = h0
            shower_c.h_top = h_top
            shower_c.N_steps = N_steps
            shower_c.model = model

    else:
        raise ValueError('The input atmosphere is not valid.')

    # Update of track, profile, fluorescence and cherenkov
    if ((theta == shower.theta) and (az == shower.az) and (x0 == shower.x0)
            and (y0 == shower.y0)):  # If the same track is also used
        shower_c.track = shower.track  # New alias
    else:
        # Otherwise a new track is generated from the input parameters
        shower_c.track = sm.Track(theta, None, az, x0, y0, atmosphere)

    # If the same profile is used
    if ((E == shower.E) and (theta == shower.theta)
            and (prf_model == shower.prf_model) and (X_max == shower.X_max)
            and (X0_GH == shower.X0_GH) and (lambda_GH == shower.lambda_GH)):
        # New alias for profile, fluorescence and cherenkov
        shower_c.profile = shower.profile
        shower_c.fluorescence = shower.fluorescence
        shower_c.cherenkov = shower.cherenkov
    else:   # Otherwise a new profile is generates from the input parameters
        shower_c.profile = sm.Profile(E, theta, None, prf_model, X_max, X0_GH,
                                      lambda_GH, atmosphere)
        # Fluorescence emission
        shower_c.fluorescence = shower_c.profile.Fluorescence()
        # Cherenkov emission
        shower_c.cherenkov = shower_c.profile.Cherenkov()

    return shower_c

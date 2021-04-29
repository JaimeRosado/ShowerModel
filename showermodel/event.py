# coding: utf-8

import numpy as np
import math
import showermodel as sm
import matplotlib as mpl
import matplotlib.pyplot as plt


# Constructor #################################################################
def Event(observatory, shower, atm_trans=True, tel_eff=True, **kwargs):
    """
    Construct an Event object from a shower and an observatory.
    
    The Event objet contains the signal produced by the shower in each
    telescope of the observatory.

    Parameters
    ----------
    observatory : Observatory object (may be a Grid object).
    shower : Shower object.
    atm_trans : bool, default True
        Include the atmospheric transmision to transport photons.
    tel_eff : bool, default True
        Include the telescope efficiency to calculate the signals.
        If False, 100% efficiency is assumed for a given wavelength interval.
    **kwargs {wvl_ini, wvl_fin, wvl_step}
        These parameters will be passed to the Signal constructor to modify
        the wavelength interval when tel_eff==False. If None, the wavelength
        interval defined in each telescope is used.

    Returns
    -------
    event : Event object.
    """
    from .observatory import _Observatory, _Grid
    from .telescope import _Telescope
    from .shower import _Shower
    if not isinstance(shower, _Shower):
        observatory, shower = (shower, observatory)
        if not isinstance(shower, _Shower):
            raise ValueError('The input shower is not valid')

    if isinstance(observatory, (_Telescope, _Observatory)):
        if isinstance(observatory, _Grid):
            event = _GridEvent()
            event.grid = observatory
        else:
            event = _Event()
            if isinstance(observatory, _Telescope):
                telescope = observatory
                event.observatory = _Observatory()
                event.observatory.append(telescope)
            else:
                event.observatory = observatory
    else:
        raise ValueError('The input observatory is not valid')

    event.shower = shower
    event.atmosphere = shower.atmosphere
    event.track = shower.track
    event.profile = shower.profile
    event.cherenkov = shower.cherenkov
    event.fluorescence = shower.fluorescence

    event.atm_trans = atm_trans
    event.tel_eff = tel_eff

    event.projections = []
    event.signals = []
    for telescope in observatory:
        projection = sm.Projection(telescope, event.track)
        event.projections.append(projection)
        signal = sm.Signal(
            telescope, shower, projection, atm_trans, tel_eff, **kwargs)
        event.signals.append(signal)
        
    event.images = None

    return event


# Class #######################################################################
class _Event():
    """
    An Event object contains the characteristics of a shower, an observatory and
    the signal produced by the shower in each telescope.

    Use Event to construct an Event object.

    Attributes
    ----------
    event_type : {'Event', 'GridEvent'} or new name
        Name of subclass of Event. Presently, only the parent class Event and
        the subclass GridEvent are available. More subclasses to be implemented.
    shower : Shower object.
    track : Track object.
    profile : Profile object.
    fluorescence : Fluorescence object.
    cherenkov : Cherenkov object.
    atmosphere : Atmosphere object.
    observatory : Observatory object.
    grid : Grid object (only for GridEvent objects)
        It replaces observatory for GridEvent objects.
    projections : List of Projection objects.
    signals : List of Signal objects.
    images : List of Image objects
        Only available if generated via the method make_images.
    atm_trans : bool
        True if the atmospheric transmision is included.
    tel_eff : bool
        True if the telescope efficiency is included.

    Methods
    -------
    show_projection : Show the projection of the shower track viewed by a
        telescope in both zenith and camera projections.
    show_profile : Show the shower profile, both number of charged particles
        and energy deposit, as a function of slant depth.
    show_light_production : Show the production of both Cherenkov and
        fluorescence photons in the 290 - 430 nm range as a function of
        slant depth.
    show_signal : Show the signal evolution as a function of both time and beta
        angle (relative to the shower axis direction) for a chosen telescope of
        the observatory.
    show_geometry2D : Show the shower track together with the telescope
        positions in a 2D plot.
    show_geometry3D : Show the shower track together with the telescope
        positions in a 3D plot.
    make_images : Generate shower images.
    show_images : Show shower images (if already exist).
    """
    event_type = None

    def show_projection(self, tel_index=0, shower_size=True, axes=True,
                        max_theta=30., X_mark='X_max'):
        """
        Show the projection of the shower track viewed by a chosen telescope in
        both horizontal and FoV coordinates systems.

        Parameters
        ----------
        tel_index : int
            Index of the chosen telescope of the observatory.
        shower_size : bool, default True
            Make the radii of the shower track points proportional to the
            shower size.
        axes : bool, default True
            Show the axes of both frames of reference.
        max_theta : float, default 30 degrees
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float or None
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure, default to X_max. If X_mark is set to None,
            no mark is included.

        Returns
        -------
        (ax1, ax2) : PolarAxesSubplot objects.
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        projection = self.projections[tel_index]
        profile = self.profile
        from ._tools import show_projection
        return show_projection(projection, profile, shower_size, axes,
                               max_theta, X_mark)

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
        return self.shower.show_light_production()

    def show_signal(self, tel_index=0):
        """
        Show the signal evolution as a function of both time and beta angle
        (relative to the shower axis direction)
        for a chosen telescope of the observatory.

        Parameters
        ----------
        tel_index : int, default 0
            Index of the chosen telescope of the observatory.

        Returns
        -------
        (ax1, ax2) : AxesSubplot objects.
        """
        signal = self.signals[tel_index]
        return signal.show()

    def append_telescope(self, telescope):
        """Append a telescope to the observatory and generate the corresponding
        projection and signal."""
        self.observatory.append(telescope)
        self.observatory.N_tel += 1
        projection = telescope.Projection(self.shower)
        self.projections.append(projection)
        self.signals.append(telescope.Signal(self.shower, projection,
                                             self.atm_trans))

    def show_geometry2D(self, x_min=-1., x_max=1., y_min=-1, y_max=1.,
                        X_mark='X_max', shower_size=True, signal_size=True,
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
        signal_size : bool
            Make the radii of the telescope position points proportional to
            the signal.
        tel_index : bool
            Show the telescope indexes together the telescope position points.

        Returns
        -------
        ax : AxesSubplot object.
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        observatory = (self.grid if self.event_type == 'GridEvent'
                       else self.observatory)
        from ._tools import show_geometry
        return show_geometry(self, observatory, '2d', x_min, x_max, y_min,
                             y_max, X_mark, shower_size, signal_size,
                             tel_index, False, False)

    def show_geometry3D(self, x_min=-1., x_max=1., y_min=-1, y_max=1.,
                        X_mark='X_max', shower_size=True, signal_size=True,
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
            Make the radii of the shower track points proportional to the
            shower size.
        signal_size : bool, default True
            Make the radii of the telescope position points proportional to
            the signal.
        xy_proj : bool, default True
            Show the xy projection of the shower track.
        pointing : bool, default False
            Show the telescope axes.

        Returns
        -------
        ax : Axes3DSubplot object.
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        observatory = (self.grid if self.event_type == 'GridEvent'
                       else self.observatory)
        from ._tools import show_geometry
        return show_geometry(self, observatory, '3d', x_min, x_max, y_min,
                             y_max, X_mark, shower_size, signal_size, False,
                             xy_proj, pointing)

    def show_distribution(self, grid=None, size_x=2., size_y=2., N_x=10,
                          N_y=10, atm_trans=None, tel_eff=None, **kwargs):
        """
        Make a GridEvent object and show the distribution of photons
        (or photoelectrons) per m$^2$ in an either 1D or 2D plot, depending on
        the grid dimensions.

        Parameters
        ----------
        grid : Grid object
            If None, a new Grid object is generated from the specified
            dimensions and the characteristics of the telescope with tel_index=0
            of the observatory. If given, size_x, size_y, N_x, N_y are not used.
        size_x : float
            Size of the grid in km across the x direction.
        size_y : float
            Size of the grid in km across the y direction.
        N_x : int
            Number of cells across the x direction.
        N_y : int
            Number of cells across the y direction.
        atm_trans : bool, default True
            Include the atmospheric transmision to transport photons. If None,
            this option is set to be the same as the original Event object.
        tel_eff : bool, default True
            Include the telescope efficiency to calculate the signals. If None,
            this option is set to be the same as the original Event object.
        **kwargs {wvl_ini, wvl_fin, wvl_step}
            These parameters will be passed to the Signal constructor to modify
            the wavelength interval when tel_eff==False. If None, the wavelength
            interval the grid telescopes is used.

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
                observatory = self.observatory
                telescope = observatory[0]  # tel_index=0 is used as reference
                tel_type = telescope.tel_type
                x_c = observatory.x_c
                y_c = observatory.y_c
                z_c = observatory.z_c
                theta = telescope.theta
                alt = telescope.alt
                az = telescope.az
                grid =sm.Grid(telescope, tel_type, x_c, y_c, z_c, theta, alt,
                               az, size_x, size_y, N_x, N_y)
            else:
                raise ValueError('The input grid is not valid')

        # Default values from the original event
        atm_trans = self.atm_trans if atm_trans is None else atm_trans
        tel_eff = self.tel_eff if tel_eff is None else tel_eff
        signal = self.signals[0]  # tel_index=0 is used as reference
        kwargs['wvl_ini'] = kwargs.get('wvl_ini', signal.wvl_ini)
        kwargs['wvl_fin'] = kwargs.get('wvl_fin', signal.wvl_fin)
        kwargs['wvl_step'] = kwargs.get('wvl_step', signal.wvl_step)

        grid_event = sm.Event(grid, self.shower, atm_trans, tel_eff, **kwargs)
        return grid_event.show_distribution()

    def make_images(self, lat_profile=True, NSB=40.):
        """
        Generate a time-varying shower image for each telescope assuming a
        circular camera with square pixels of same solid angle. The list of
        images is stored in the attribute images of the Event object.

        Parameters
        ----------
        lat_profile : book
            Use a NKG lateral profile to spread the signal. If False,
            a linear shower is assumed.
        NSB : float
            Night sky background in MHz/m$^2$/deg$^2$.

        Returns
        -------
        images : List of Image objects.

        See also
        --------
        Image : Constructor of Image object.
        """
        images = [sm.Image(signal, lat_profile=lat_profile, NSB=NSB)
                  for signal in self.signals]
        self.images = images
        return images

    def show_images(self, col=5, size=2):
        """
        Show subplots of shower images (if already exist). Each subplot is
        labelled with the telescope id.
        
        Parameters
        ----------
        col : int
            Number of columns of the figure. Default to 5.
        size : float, default 2
            Subplot size in cm.
        """
        if self.images is None:
            raise ValueError(
                'Images must be generated first via make_images method.')
        rows = math.ceil(len(self.observatory)/col)
        fig, axes = plt.subplots(rows, col, figsize=(col*size, rows*size))
        plt.tight_layout()
        N = len(self.observatory)
        for tel, ax in enumerate(axes.flatten()):
            if tel < N:
                ax = self.images[tel].show(ax=ax)
                ax.set_title(tel)
        #plt.show()


class _GridEvent(_Event):
    event_type = 'GridEvent'

    def show_distribution(self):  # Overwrite the method of the parent class
        """
        Show the distribution of photons (or photoelectrons) per m$^2$ in an
        either 1D or 2D plot, depending on the grid dimensions.
        """
        return _show_distribution(self)


# Auxiliary functions #########################################################
def _show_distribution(grid_event):
    """
    Show the distribution of photons (or photoelectrons) per m$^2$ in an either
    1D or 2D plot, depending on the grid dimensions.
    """
    grid = grid_event.grid
    # Detection area of the telescope, default to grid cell area.
    area = grid.tel_area

    signal_cher = np.array(
        [signal.Npe_cher_sum for signal in grid_event.signals]) / area
    signal_fluo = np.array(
        [signal.Npe_fluo_sum for signal in grid_event.signals]) / area
    signal_total = signal_cher + signal_fluo

    signal_max = max(signal_cher.max(), signal_fluo.max())
    signal_min = min(signal_cher.min(), signal_fluo.min())
    if signal_max == 0.:
        print('The shower track is outside the field of view of telescopes.')
        return
    if signal_min == 0.:
        signal_min = signal_max / 1000.

    N_x = grid.N_x
    N_y = grid.N_y
    x_c = grid.x_c
    y_c = grid.y_c
    size_x = grid.size_x
    size_y = grid.size_y
    if (N_x > 1) and (N_y > 1):    # 2D grid
        # 2D distributions
        signal_cher = signal_cher.reshape(N_y, N_x)
        signal_fluo = signal_fluo.reshape(N_y, N_x)

        # Image frame
        extent = (x_c-size_x/2., x_c+size_x/2., y_c-size_y/2., y_c+size_y/2.)

        # Image plots with color map in logaritmic scale
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4),
                                       constrained_layout=True)
        ax1.imshow(signal_cher,
                   norm=mpl.colors.LogNorm(vmin=signal_min, vmax=signal_max),
                   extent=extent)  # cmap=viridis
        psm = ax2.imshow(signal_fluo,
                         norm=mpl.colors.LogNorm(
                             vmin=signal_min, vmax=signal_max),
                         extent=extent)

        ax1.set_title('Cherenkov')
        ax1.axes.xaxis.set_label_text('x (km)')
        ax1.axes.yaxis.set_label_text('y (km)')
        ax2.set_title('Fluorescence')
        ax2.axes.xaxis.set_label_text('x (km)')
        ax2.axes.yaxis.set_label_text('y (km)')

        # Color bar attached to second plot
        cbar = fig.colorbar(psm)
        if grid_event.tel_eff:  # With telescope efficiency
            cbar.ax.set_ylabel('Photoelectrons / m$^2$')
        else:        # Without telescope efficiency
            cbar.ax.set_ylabel('Photons / m$^2$')

        return ax1, ax2, cbar

    else:  # 1D grid
        # Logarithmic scale plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plt.yscale('log')

        # Telescope positions
        coords = [(telescope.x, telescope.y) for telescope in grid]
        x, y = zip(*coords)

        # Determine if the grid is across the x axis or y axis
        if N_x == 1:
            x = y
            ax.axes.xaxis.set_label_text('y (km)')
        else:
            ax.axes.xaxis.set_label_text('x (km)')

        if grid_event.tel_eff:  # With telescope efficiency
            ax.axes.yaxis.set_label_text('Photoelectrons / m$^2$')
        else:        # Without telescope efficiency
            ax.axes.yaxis.set_label_text('Photons / m$^2$')

        ax.plot(x, signal_cher, 'r--', label='Cherenkov')
        ax.plot(x, signal_fluo, 'b--', label='Fluorescence')
        ax.plot(x, signal_total, 'k', label='Total')
        ax.legend()

        return ax

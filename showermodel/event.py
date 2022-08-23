# coding: utf-8

import numpy as np
import showermodel as sm
import showermodel.constants as ct
import matplotlib as mpl
import matplotlib.pyplot as plt

# Default values for Event
_Event__atm_trans = ct.config['Signal']['atm_trans']
_Event__tel_eff = ct.config['Signal']['tel_eff']
_Event__wvl_ini = ct.config['Signal']['wvl_ini']
_Event__wvl_fin = ct.config['Signal']['wvl_fin']
_Event__wvl_step = ct.config['Signal']['wvl_step']

# Default values for Grid
_Grid__obs_name = ct.config['Grid'].get('obs_name') # optional parameter
_Grid__size_x = ct.config['Grid']['size_x']
_Grid__size_y = ct.config['Grid']['size_y']
_Grid__N_x = ct.config['Grid']['N_x']
_Grid__N_y = ct.config['Grid']['N_y']

# Default values for Image
_Image__lat_profile = ct.config['Image']['lat_profile']
_Image__NSB = ct.config['Image']['NSB']

# Class #######################################################################
class Event():
    """
    Contain the characteristics of a shower detected by an observatory.

    The signal produced by the shower in each telescope of the observatory is
    stored in a list. The shower, the observatory, etc. are stored as object
    attributes.

    Parameters
    ----------
    observatory : Observatory, mandatory
        Observatory that observes the shower.
    shower : Shower, mandatory
        Shower to be observed.
    atm_trans : bool, default True
        Include the atmospheric transmission to transport photons.
    tel_eff : bool, default True
        Include the telescope efficiency to calculate the signals.
        If False, 100% efficiency is assumed for a given wavelength interval.
    wvl_ini : float, default 290
        Initial wavelength in nm of the interval to calculate the signal when
        tel_eff==False.
    wvl_fin : float, default 430
        Final wavelength in nm of the interval to calculate the signal when
        tel_eff==False.
    wvl_step : float, default 3
        Discretization step in nm of the interval to calculate the signal when
        tel_eff==False.

    Attributes
    ----------
    shower : Shower
    track : Track
    profile : Profile
    fluorescence : Fluorescence
    cherenkov : Cherenkov
    atmosphere : Atmosphere
    observatory : Observatory
    projections : list
        List of Projection objects, one per telescope.
    signals : list
        List of Signal objects, one per telescope.
    images : list or None
        List of Image objects, one per telescope.
        Only available if generated via the method make_images.
    atm_trans : bool
        True if the atmospheric transmission is included.
    tel_eff : bool
        True if the telescope efficiency is included.

    Methods
    -------
    show_projection()
        Show the projection of the shower track viewed by a telescope.
    show_profile()
        Show the shower profile as a function of slant depth.
    show_light_production()
        Show the production of both Cherenkov and fluorescence photons as a
        function of slant depth.
    show_signal()
        Show the signal evolution for a chosen telescope of the observatory.
    show_geometry2D()
        Show the shower track and the telescope positions in a 2D plot.
    show_geometry3D()
        Show the shower track and the telescope positions in a 3D plot.
    show_distribution()
        Make a GridEvent object and show the distribution of photons
        per m^2 in a 1D or 2D plot.
    make_images()
        Generate shower images.
    show_images()
        Show shower images (if already exist).
    """

    def __init__(self, observatory, shower, atm_trans=__atm_trans,
                 tel_eff=__tel_eff, wvl_ini=__wvl_ini, wvl_fin=__wvl_fin,
                 wvl_step=__wvl_step):
        _event(self, observatory, shower, atm_trans, tel_eff,
               wvl_ini, wvl_fin, wvl_step)

    def show_projection(self, tel_index=0, shower_Edep=True, axes=True,
                        max_theta=30., X_mark='X_max'):
        """
        Show the projection of the shower track viewed by a chosen telescope in
        both horizontal and FoV coordinates systems.

        Parameters
        ----------
        tel_index : int, default 0
            Index of the chosen telescope of the observatory.
        shower_Edep : bool, default True
            Make the radii of the shower track points proportional to the
            energy deposited in each step length.
        axes : bool, default True
            Show the axes of both frames of reference.
        max_theta : float, default 30
            Maximum offset angle in degrees relative to the telescope
            pointing direction.
        X_mark : float
            Reference slant depth in g/cm^2 of the shower track to be
            marked in the figure. If set to None, no mark is included.
            By default, the mark is placed at X_max.

        Returns
        -------
        (ax1, ax2) : PolarAxesSubplot
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        projection = self.projections[tel_index]
        profile = self.profile
        from ._tools import show_projection
        return show_projection(projection, profile, shower_Edep, axes,
                               max_theta, X_mark)

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
        (ax1, ax2) : AxesSubplot
        """
        signal = self.signals[tel_index]
        return signal.show()

    def append_telescope(self, telescope):
        """
        Append a telescope to the observatory and generate the corresponding
        projection and signal.
        
        Parameters
        ----------
        telescope : Telescope, mandatory
            Telescope to be appended.
        """
        self.observatory.append(telescope)
        projection = telescope.Projection(self.shower)
        self.projections.append(projection)
        self.signals.append(telescope.Signal(self.shower, projection,
                                             self.atm_trans))

    def show_geometry2D(self, x_min=-1., x_max=1., y_min=-1, y_max=1.,
                        X_mark='X_max', shower_Edep=True, signal_size=True,
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
        signal_size : bool, default True
            Make the radii of the telescope position points proportional to
            the signal.
        tel_index : bool, default False
            Show the telescope indexes together the telescope position points.

        Returns
        -------
        ax : AxesSubplot
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        observatory = self.observatory  # observatory==grid for GridEvent
        from ._tools import show_geometry
        return show_geometry(self, observatory, '2d', x_min, x_max, y_min,
                             y_max, X_mark, shower_Edep, signal_size,
                             tel_index, False, False)

    def show_geometry3D(self, x_min=-1., x_max=1., y_min=-1, y_max=1.,
                        X_mark='X_max', shower_Edep=True, signal_size=True,
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
        signal_size : bool, default True
            Make the radii of the telescope position points proportional to
            the signal.
        xy_proj : bool, default True
            Show the xy projection of the shower track.
        pointing : bool, default False
            Show the telescope axes.

        Returns
        -------
        ax : Axes3DSubplot
        """
        if X_mark == 'X_max':
            X_mark = self.shower.X_max
        observatory = self.observatory
        from ._tools import show_geometry
        return show_geometry(self, observatory, '3d', x_min, x_max, y_min,
                             y_max, X_mark, shower_Edep, signal_size, False,
                             xy_proj, pointing)

    def show_distribution(self, grid=None, tel_index=0, size_x=_Grid__size_x,
                          size_y=_Grid__size_y, N_x=_Grid__N_x, N_y=_Grid__N_y,
                          **kwargs):
        """
        Make a new Event from a Grid object based on one of the telescopes of
        the observatory and show the distribution of photons (or
        photoelectrons) per m^2 in this grid. Results are shown in an either 1D
        or 2D plot, depending on the grid dimensions.

        Parameters
        ----------
        tel_index : int, default 0
            Index of the telescope used to generate the grid when grid==None.
            The grid is centered at the telescope location and the pointing
            direction is set to be the same.
        size_x : float, defaut 2
            Size of the grid in km across the x direction.
        size_y : float, default 2
            Size of the grid in km across the y direction.
        N_x : int, default 10
            Number of cells across the x direction.
        N_y : int, default 10
            Number of cells across the y direction.
<<<<<<< Updated upstream
        atm_trans : bool, default True
            Include the atmospheric transmission to transport photons. If None,
            this option is set to be the same as the original Event object.
        tel_eff : bool, default True
            Include the telescope efficiency to calculate the signals. If None,
            this option is set to be the same as the original Event object.
        **kwargs : {wvl_ini, wvl_fin, wvl_step}
            These parameters will be passed to the Signal constructor to modify
            the wavelength interval when tel_eff==False. If None, the wavelength
            interval the grid telescopes is used.
=======
        atm_trans : bool
            Include the atmospheric transmision to transport photons.
            By default, this option is set to be the same as the Event object.
        tel_eff : bool
            Include the telescope efficiency to calculate the signals.
            By default, this option is set to be the same as the Event object.
        wvl_ini : float
            Initial wavelength in nm of the interval to calculate the signal
            when tel_eff==False. By default, this parameter is set to be the
            same as the Event object.
        wvl_fin : float
            Final wavelength in nm of the interval to calculate the signal when
            tel_eff==False. By default, this parameter is set to be the same as
            the Event object.
        wvl_step : float
            Discretization step in nm of the interval to calculate the signal
            when tel_eff==False. By default, this parameter is set to be the
            same as the Event object.
>>>>>>> Stashed changes

        Returns
        -------
        grid_event : Event
        ax : AxesSubplot 
            If 1D grid.
        (ax1, ax2, cbar) : AxesSubplot and Colorbar
            If 2D grid.
        """
<<<<<<< Updated upstream
        if not isinstance(grid, sm.Grid):
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
                grid = sm.Grid(telescope, x_c, y_c, z_c, theta, alt,
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

        grid_event = GridEvent(grid, self.shower, atm_trans, tel_eff, **kwargs)
        return grid_event.show_distribution()

    def make_images(self, lat_profile=True, NSB=40.):
=======
        observatory = self.observatory
        telescope = observatory[tel_index]
        tel_type = telescope.tel_type
        x_c = telescope.x
        y_c = telescope.y
        z_c = telescope.z
        theta = telescope.theta
        alt = telescope.alt
        az = telescope.az
        obs_name = _Grid__obs_name
        grid =sm.Grid(obs_name, telescope, tel_type, x_c, y_c, z_c,
                      size_x, size_y, N_x, N_y, theta, alt, az)

        # Default values from the Event object
        atm_trans = kwargs.get('atm_trans', self.atm_trans)
        tel_eff = kwargs.get('tel_eff', self.tel_eff)
        wvl_ini = kwargs.get('wvl_ini', self.wvl_ini)
        wvl_fin = kwargs.get('wvl_fin', self.wvl_fin)
        wvl_step = kwargs.get('wvl_step', self.wvl_step)

        grid_event = Event(grid, self.shower, atm_trans, tel_eff, wvl_ini,
                           wvl_fin, wvl_step)
        return grid_event, _show_distribution(grid_event)

    def make_images(self, lat_profile=_Image__lat_profile, NSB=_Image__NSB):
>>>>>>> Stashed changes
        """
        Generate a time-varying shower image for each telescope assuming a
        circular camera with square pixels of same solid angle. The list of
        images is stored in the attribute images of the Event object.

        Parameters
        ----------
        lat_profile : bool, default True
            Use a NKG lateral profile to spread the signal. If False,
            a linear shower is assumed.
        NSB : float, default 40
            Night sky background in MHz/m^2/deg^2.

        Returns
        -------
        images : list
            List of Image objects.

        See also
        --------
        Image : Constructor of Image object.
        """
        # N_pix and int_time are not allowed to be changed
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
        col : int, default 5
            Number of columns of the figure.
        size : float, default 2
            Subplot size in cm.
        """
        if self.images is None:
            raise ValueError(
                'Images must be generated first via make_images method.')
        rows = int(np.ceil(len(self.observatory)/col))
        fig, axes = plt.subplots(rows, col, figsize=(col*size, rows*size))
        plt.tight_layout()
        N = len(self.observatory)
        for tel, ax in enumerate(axes.flatten()):
            if tel < N:
                ax = self.images[tel].show(ax=ax)
                ax.set_title(tel)
        #plt.show()


# Constructor #################################################################
def _event(event, observatory, shower, atm_trans, tel_eff,
           wvl_ini, wvl_fin, wvl_step):
    """
    Construct an Event object from a shower and an observatory.
    
    The Event objet contains the signal produced by the shower in each
    telescope of the observatory.

    Parameters
    ----------
    event : Event
        Event to be generated.
    observatory : Observatory
        Observatory that observes the shower.
    shower : Shower
        Shower to be observed.
    atm_trans : bool, default True
        Include the atmospheric transmission to transport photons.
    tel_eff : bool, default True
        Include the telescope efficiency to calculate the signals.
        If False, 100% efficiency is assumed for a given wavelength interval.
    wvl_ini, wvl_fin, wvl_step : float
        Wavelength interval to calculate the signal when tel_eff==False.
    """
    if not isinstance(shower, sm.Shower):
        observatory, shower = (shower, observatory)
        if not isinstance(shower, sm.Shower):
            raise TypeError('The input shower is not a Shower object.')

    if isinstance(observatory, (sm.Telescope, sm.Observatory)):
        if isinstance(observatory, sm.Telescope):
            telescope = observatory
            event.observatory = sm.Observatory(telescope)        
        else:
            event.observatory = observatory
    else:
        raise TypeError('The input observatory is not an Observatory object.')

    event.shower = shower
    event.atmosphere = shower.atmosphere
    event.track = shower.track
    event.profile = shower.profile
    event.cherenkov = shower.cherenkov
    event.fluorescence = shower.fluorescence

    event.atm_trans = atm_trans
    event.tel_eff = tel_eff
    # Even if tel_eff==True, wvl parameters are stored
    # They may be used in Event.show_distribution with tel_eff=False
    event.wvl_ini = wvl_ini
    event.wvl_fin = wvl_fin
    event.wvl_step = wvl_step

    event.projections = []
    event.signals = []
    for telescope in observatory:
        projection = sm.Projection(telescope, event.track)
        event.projections.append(projection)
        signal = sm.Signal(telescope, shower, projection, atm_trans,
                           tel_eff, wvl_ini, wvl_fin, wvl_step)
        event.signals.append(signal)
    
    event.images = None


# Auxiliary functions #########################################################
def _show_distribution(grid_event):
    """
    Show the distribution of photons (or photoelectrons) per m^2 in an either
    1D or 2D plot, depending on the grid dimensions.
    """
    grid = grid_event.observatory
    if not isinstance(grid, sm.Grid): # error if grid is not Grid
        raise ValueError("The observatory should be a Grid object.")

    signal_cher = np.array(
        [signal.Npe_cher_sum / signal.telescope.area
         for signal in grid_event.signals])
    signal_fluo = np.array(
        [signal.Npe_fluo_sum / signal.telescope.area
         for signal in grid_event.signals])
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

        # Image plots with color map in logarithmic scale
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

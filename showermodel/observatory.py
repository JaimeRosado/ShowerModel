# coding: utf-8

import numpy as np
import showermodel.constants as ct
import matplotlib.pyplot as plt
from .telescope import Telescope, _pointing

# Classes #####################################################################

# Default values for Observatory
_Observatory__obs_name = ct.config['Observatory'].get('obs_name') # optional
_Observatory__theta = ct.config['Telescope']['theta']
_Observatory__alt = ct.config['Telescope'].get('alt') # optional parameter
_Observatory__az = ct.config['Telescope']['az']

class Observatory(list):
    """
    List of telescopes.

    The characteristics of the observatory are stored in attributes.

    Note: Attributes inherited from Telescope (i.e., tel_type, tel_apert,
    tel_area and tel_N_pix) are not updated when telescopes are modified
    or appended.

    Parameters
    ----------
    *telescopes : Telescope, mandatory
        List of telescopes objects to be included.
    obs_name : str or None, default None
        Name given to the observatory.

    Attributes
    ----------
    obs_name : str or None
        Name given to the observatory.
    N_tel : int
        Number of telescopes.
    tel_type : str or None
        Telescope type. Only defined if all the telescopes are of the same
        type.
    theta : float or None
        Zenith angle in degrees of the observatory pointing direction. Only
        defined if all the telescopes point to the same direction.
    alt : float or None
        Altitude in degrees of the observatory pointing direction. Only
        defined if all the telescopes point to the same direction.
    az : float or None
        Azimuth angle (from north, clockwise) in degrees of the observatory
        pointing direction. Only defined if all the telescopes point to the
        same direction.

    Methods
    -------
    show()
        Show the telescope positions and indexes in a 2D plot.
    append()
        Append a telescope to the observatory.
    set_pointing()
        Set pointings of all the telescopes.
    """
    def __init__(self, *telescopes, obs_name=__obs_name):
        if not np.all([isinstance(tel, Telescope) for tel in telescopes]):
            raise TypeError("Input telescopes are not of type Telescope.")
        super().__init__([*telescopes])
        self.obs_name = obs_name



    @property
    def N_tel(self):
        self._N_tel = len(self)
        return self._N_tel

    @property
    def tel_type(self):
        tel_type = self[0].tel_type
        if np.all([tel.tel_type==tel_type for tel in self]):
            self._tel_type = tel_type
        else:
            self._tel_type = None
        return self._tel_type

    @property
    def theta(self):
        self._check_pointing()
        return self._theta

    @theta.setter
    def theta(self, new_theta):
        self.set_pointing(new_theta, None, self.az)

    @property
    def alt(self):
        self._check_pointing()
        return self._alt

    @alt.setter
    def alt(self, new_alt):
        self.set_pointing(0., new_alt, self.az)

    @property
    def az(self):
        self._check_pointing()
        return self._az

    @az.setter
    def az(self, new_az):
        self.set_pointing(self.theta, None, new_az)

    def show(self):
        """
        Show the telescope positions and indexes of the observatory in a
        2D plot.
        """
        return _show(self)

    def append(self, telescope):
        """
        Append a new telescope and increase N_tel.

        Parameters
        ----------
        telescope : Telescope
            Telescope to be added.
        """
        super().append(telescope)
<<<<<<< Updated upstream
        self.N_tel += 1
=======

    def _check_pointing(self):
        theta = self[0].theta
        az = self[0].az
        if np.all([(tel.theta==theta) & (tel.az==az) for tel in self]):
            self._theta = theta
            self._alt = 90. - theta
            self._az = az
        else:
            self._theta = None
            self._alt = None
            self._az = None

    def set_pointing(self, theta=__theta, alt=__alt, az=__az):
        """
        Set pointings of all the telescopes.

        Parameters
        ----------
        theta : float, default 0
            Zenith angle in degrees of the telescope pointing direction.
        alt : float, default None
            Altitude in degrees of the telescope pointing direction.
            If None, theta is used. If given, theta is overwritten.
        az : float, default 0
            Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing direction.
        """
        if alt is None:
            alt = 90. - theta
        else:
            theta = 90. - alt
        for tel in self:
            _pointing(tel, theta, alt, az)
        self._theta = theta
        self._alt = alt
        self._az = az

# Default values for Array25
_Array25__obs_name = ct.config['Array25'].get('obs_name') # optional parameter
_Array25__tel_type = ct.config['Array25']['tel_type']
_Array25__x_c = ct.config['Array25']['x_c']
_Array25__y_c = ct.config['Array25']['y_c']
_Array25__z_c = ct.config['Array25']['z_c']
_Array25__R = ct.config['Array25']['R']
_Array25__rot_angle = ct.config['Array25']['rot_angle']
_Array25__theta = _Observatory__theta
_Array25__alt = _Observatory__alt
_Array25__az = _Observatory__az
>>>>>>> Stashed changes


class Array25(Observatory):
    """
    Array of 25 telescopes similar to the layout of MST telescopes of CTA.

    The pointing directions of all the telescopes are set equally, but they can
    be modified individually (along with other properties) later on.

    The telescope index is sorted first by radius and then by azimuth, so that
    tel_index=0 corresponds to the central telescope.

    Parameters
    ----------
    obs_name : str, default 'Array25'
        Name given to the observatory.
    telescope : Telescope, default None
        Telescope object to be used to construct the observatory. If None, the
        given tel_type telescope is used.
    tel_type : str, default 'IACT'
        Type of telescope to be used when telescope==None.
    x_c : float, default 0
        East coordinate in km of the center of the array.
    y_c : float, defatul 0
        North coordinate in km of the center of the array.
    z_c : float, default 0
        Height of the array in km above ground level.
    R : float, default 341
        Radius in km of the array.
    rot_angle : float, default 0
        Rotation angle in degrees of the array (clockwise).
    theta : float, default 0
        Zenith angle in degrees of the telescope pointing directions.
    alt : float, default None
        Altitude in degrees of the telescope pointing directions. If None,
        theta is used. If given, theta is overwritten.
    az : float, default 0
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing directions.

    Attributes
    ----------
    obs_name : str
        Name given to the observatory.
    N_tel : int
        Number of telescopes.
    x_c : float
        East coordinate in km of the center of the array.
    y_c : float
        North coordinate in km of the center of the array.
    z_c : float
        Height of the array in km above ground level.
    R : float
        Radius in km of the array.
    rot_angle : float
        Rotation angle in degrees of the array (clockwise).
    tel_type : str or None
        Telescope type. Only defined if all the telescopes are of the same
        type.
    theta : float or None
        Zenith angle in degrees of the observatory pointing direction. Only
        defined if all the telescopes point to the same direction.
    alt : float or None
        Altitude in degrees of the observatory pointing direction. Only
        defined if all the telescopes point to the same direction.
    az : float or None
        Azimuth angle (from north, clockwise) in degrees of the observatory
        pointing direction. Only defined if all the telescopes point to the
        same direction.
    """
    def __init__(self, obs_name=__obs_name, telescope=None,
                 tel_type=__tel_type, x_c=__x_c, y_c=__y_c, z_c=__z_c,
                 R=__R, rot_angle=__rot_angle, theta=__theta, alt=__alt,
                 az=__az):
        self.obs_name = obs_name
        _array25(self, telescope, tel_type, x_c, y_c, z_c, R, rot_angle,
                theta, alt, az)

    @property
    def x_c(self):
        return self._x_c

    @property
    def y_c(self):
        return self._y_c

    @property
    def z_c(self):
        return self._z_c

    @property
    def R(self):
        return self._R

    @property
    def size_x(self):
        return self._size_x

    @property
    def size_y(self):
        return self._size_y

    @property
    def rot_angle(self):
        return self._rot_angle

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
_Grid__theta = _Observatory__theta
_Grid__alt = _Observatory__alt
_Grid__az = _Observatory__az

class Grid(Observatory):
    """
    Rectangular grid of telescopes across the x and y directions.

    The pointing directions of all the telescopes are set equally, but they can
    be modified individually (along with other properties) later on.

    The telescope index is sorted first by y (from max to min) and then by x
    (from min to max), so that tel_index=0 corresponds to the telescope placed
    at the corner with minimum x and maximum y.

    Parameters
    ----------
    obs_name : str, default 'Grid'
        Name given to the observatory.
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
        Height of the grid in km above ground level.
    size_x : float, defaut 2
        Size of the grid in km along the x direction.
    size_y : float, default 2
        Size of the grid in km along the y direction.
    N_x : int, default 10
        Number of cells along the x direction.
    N_y : int, default 10
        Number of cells along the y direction.
    theta : float, default 0
        Zenith angle in degrees of the telescope pointing directions.
    alt : float, default None
        Altitude in degrees of the telescope pointing directions. If None,
        theta is used. If given, theta is overwritten.
    az : float, default 0
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing directions.

    Attributes
    ----------
    obs_name : str
        Name given to the observatory.
    N_tel : int
        Number of cells.
    x_c : float
        East coordinate in km of the center of the grid.
    y_c : float
        North coordinate in km of the center of the grid.
    z_c : float
        Height of the grid in km above ground level.
    size_x : float
        Size of the grid in km across the x direction.
    size_y : float
        Size of the grid in km across the y direction.
    N_x : int
        Number of cells across the x direction.
    N_y : int
        Number of cells across the y direction.
    cell_area : float
        Cell area in m^2.
    tel_type : str or None
        Telescope type. Only defined if all the telescopes are of the same
        type.
    theta : float or None
        Zenith angle in degrees of the observatory pointing direction. Only
        defined if all the telescopes point to the same direction.
    alt : float or None
        Altitude in degrees of the observatory pointing direction. Only
        defined if all the telescopes point to the same direction.
    az : float or None
        Azimuth angle (from north, clockwise) in degrees of the observatory
        pointing direction. Only defined if all the telescopes point to the
        same direction.
    """
    def __init__(self, obs_name=__obs_name, telescope=None,
                 tel_type=__tel_type, x_c=__x_c, y_c=__y_c, z_c=__z_c,
                 size_x=__size_x, size_y=__size_y, N_x=__N_x, N_y=__N_y,
                 theta=__theta, alt=__alt, az=__az):
        self.obs_name = obs_name
        _grid(self, telescope, tel_type, x_c, y_c, z_c, size_x, size_y, N_x,
              N_y, theta, alt, az)

    @property
    def x_c(self):
        return self._x_c

    @property
    def y_c(self):
        return self._y_c

    @property
    def z_c(self):
        return self._z_c

    @property
    def size_x(self):
        return self._size_x

    @property
    def size_y(self):
        return self._size_y

    @property
    def N_x(self):
        return self._N_x

    @property
    def N_y(self):
        return self._N_y

    @property
    def cell_area(self):
        return self._cell_area


# Constructors ################################################################
def _array25(observatory, telescope, tel_type, x_c, y_c, z_c, R, rot_angle,
             theta, alt, az):
    """
    Constructor of Array25 class.

    Parameters
    ----------
    observatory : Array25
    telescope : Telescope
        If None, the default IACT telescope is used.
    x_c : float
        East coordinate in km of the center of the array.
    y_c : float
        North coordinate in km of the center of the array.
    z_c : float
        Height of the array in km above ground level.
    R : float
        Radius in km of the array.
    rot_angle : float
        Rotation angle in degrees of the array (clockwise).
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
    alt : float
        Altitude in degrees of the telescope pointing directions. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing directions.
    """
    if isinstance(telescope, Telescope):
        telescope = telescope.copy(x=x_c, y=y_c, z=z_c, theta=theta, alt=alt,
                                   az=az)
    elif telescope is None:
        # The default telescope type is IACT
        telescope = Telescope(tel_type=tel_type, x=x_c, y=y_c, z=z_c,
                              theta=theta, alt=alt, az=az)
    else:
        raise TypeError('The input telescope is not of type Telescope.')

    observatory._tel_type = telescope.tel_type
    observatory._x_c = x_c
    observatory._y_c = y_c
    observatory._z_c = z_c
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    observatory._theta = theta
    observatory._alt = alt
    observatory._az = az
    observatory._R = R
    observatory._size_x = 2.*R
    observatory._size_y = 2.*R
    observatory._rot_angle = rot_angle

    observatory._N_tel = 0 # The append method increases N_tel
    # Central telescope
    observatory.append(telescope)

    # First circle of 4 telescopes
    for i in range(4):
        az = np.radians(rot_angle + 90. * i)
        xi = x_c + R / 3. * np.sin(az)
        yi = y_c + R / 3. * np.cos(az)
        observatory.append(telescope.copy(x=xi, y=yi))

    # Second circle of 8 telescopes:
    for i in range(8):
        az = np.radians(rot_angle + 45. * i)
        xi = x_c + R * 2. / 3. * np.sin(az)
        yi = y_c + R * 2. / 3. * np.cos(az)
        observatory.append(telescope.copy(x=xi, y=yi))

    # Third circle of 12 telescopes:
    for i in range(12):
        az = np.radians(rot_angle + 30. * i)
        xi = x_c + R * np.sin(az)
        yi = y_c + R * np.cos(az)
        observatory.append(telescope.copy(x=xi, y=yi))


def _grid(grid, telescope, tel_type, x_c, y_c, z_c, size_x, size_y, N_x, N_y,
          theta, alt, az):
    """
    Constructor of Grid class.

    Parameters
    ----------
    grid : Grid
    telescope : Telescope
        If None, the default GridElement object is used.
    tel_type : str, default 'GridElement'
        Type of telescope to be used when telescope==None.
    x_c : float
        East coordinate in km of the center of the grid.
    y_c : float
        North coordinate in km of the center of the grid.
    z_c : float
        Height of the grid in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
        Default to 0.
    alt : float
        Altitude in degrees of the telescope pointing directions. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle in degrees of the telescope pointing directions.
    size_x : float
        Size of the grid in km across the x direction.
    size_y : float
        Size of the grid in km across the y direction.
    N_x : int
        Number of cells across the x direction.
    N_y : int
        Number of cells across the y direction.
    """
    # Range of x, y values
    x = np.linspace(x_c-size_x/2., x_c+size_x/2., N_x + 1)  # From min to max
    y = np.linspace(y_c+size_y/2., y_c-size_y/2., N_y + 1)  # From max to min
    # Size of grid element
    step_x = size_x / N_x
    step_y = size_y / N_y
    # Telescope positions
    x_mid = x[1:] - step_x/2.
    y_mid = y[1:] + step_y/2.
    cell_area = step_x * step_y * 1000000. # m^2

    if isinstance(telescope, Telescope):
        telescope = telescope.copy(z=z_c, theta=theta, alt=alt, az=az)
    elif telescope is None:
        # The default telescope type is GridElement
        telescope = Telescope(tel_type=tel_type, z=z_c, theta=theta, alt=alt,
                              az=az)
    else:
        raise TypeError('The input telescope is not of type Telescope.')

    grid._tel_type = telescope.tel_type
    grid._x_c = x_c
    grid._y_c = y_c
    grid._z_c = z_c
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    grid._theta = theta
    grid._alt = alt
    grid._az = az
    grid._size_x = size_x
    grid._size_y = size_y
    grid._N_x = N_x
    grid._N_y = N_y
    grid._cell_area = cell_area

    grid._N_tel = 0
    for yi in y_mid:
        for xi in x_mid:
            grid.append(telescope.copy(x=xi, y=yi))


# Auxiliary functions #########################################################
def _show(observatory):
    """
    Show the telescope positions and indexes of the observatory in a 2D plot.
    """
    # Telescope positions
    coords = [(telescope.x, telescope.y) for telescope in observatory]
    x, y = zip(*coords)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, c='g', s=20, marker='o')  # Telescope positions in green

    for tel_index, telescope in enumerate(observatory):
        ax.annotate(tel_index, (x[tel_index], y[tel_index]))

    # ax.axis([x_min, x_max, y_min, y_max])
    ax.axes.xaxis.set_label_text('x (km)')
    ax.axes.yaxis.set_label_text('y (km)')
    return ax
    

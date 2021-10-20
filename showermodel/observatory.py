# coding: utf-8

import numpy as np
import showermodel as sm
import matplotlib.pyplot as plt

# Default values for array25
from .telescope import _x, _y, _z, _theta, _az
from .telescope import Telescope, IACT, GridElement


# Classes #####################################################################
class Observatory(list):
    """
    List of telescopes.

    The characterestics of the observatory are stored in attributes.

    Note: Attributes inherited from Telescope (i.e., tel_type, tel_apert,
    tel_area and tel_N_pix) are not updated when telescopes are modified
    or appended.

    Parameters
    ----------
    *telescopes : Telescope
        List of telescopes objects to be included.
    obs_type : str
        Name given to the observatory. Default to None.

    Attributes
    ----------
    obs_type : str
        Name given to the observatory. Default to None.
    N_tel : int
        Number of telescopes.
    x_c : float
        East coordinate in km of the center of the grid.
    y_c : float
        North coordinate in km of the center of the grid.
    z_c : float
        Height of the grid in km above ground level.
    size_x : float
        Size of the observatory in km across the x direction.
    size_y : float
        Size of the observatory in km across the y direction.
    N_x : int
        Number of cells across the x direction (only Grid objects).
    N_y : int
        Number of cells across the y direction (only Grid objects).
    cell_area : float
        Area in m^2 of one cell (only Grid objects).
    tel_type : str
        Name of the subclass of Telescope.
    tel_apert : float
        Angular diameter in degrees of the telescope FoV.
    tel_area : float
        Detection area of each telescope in m^2
        (e.g., mirror area of an IACT).
    tel_N_pix : int
        Number of camera pixels.        
    """
    # obs_type = None
    def __init__(self, *telescopes, obs_type=None):
        if not np.all([isinstance(tel, Telescope) for tel in telescopes]):
            raise ValueError("Input telescopes are not valid.")
        super().__init__([*telescopes])
        self.obs_type = obs_type
        _observatory(self)

    def show(self):
        """
        Show the telescope positions and indexes of the observatory in a 2D plot.

        """
        return _show(self)


class Array25(Observatory):
    """
    Array of 25 telescopes similar to the layout of MST telescopes of CTA-South.

    The pointing directions of all the telescopes are set equally, but they can
    be modified individually (along with other porperties) later on.

    The telescope index is sorted first by radius and then by azimuth, so that
    tel_index=0 corresponds to the central telescope.

    Parameters
    ----------
    telescope : Telescope
        If None, the default IACT object is used.
    x_c : float
        East coordinate in km of the center of the array.
    y_c : float
        North coordinate in km of the center of the array.
    z_c : float
        Height of the array in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
    alt : float
        Altitude in degrees of the telescope pointing directions. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing directions.
    R : float
        Radius in km of the array.
    rot_angle : float
        Rotation angle in degrees of the array (clockwise).

    Attributes
    ----------
    x_c : float
        East coordinate in km of the center of the array.
    y_c : float
        North coordinate in km of the center of the array.
    z_c : float
        Height of the array in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
    alt : float
        Altitude in degrees of the telescope pointing directions. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing directions.
    R : float
        Radius in km of the array.
    rot_angle : float
        Rotation angle in degrees of the array (clockwise).
    """
    obs_type = 'Array25'
    N_tel = 25

    def __init__(self, telescope=None, x_c=_x, y_c=_y, z_c=_z,
                 theta=None, alt=None, az=None, R=0.341, rot_angle=0.):
        _array25(self, telescope, x_c, y_c, z_c, theta, alt, az, R, rot_angle)


class Grid(Observatory):
    """
    Rectangular grid of telescopes across the x and y directions.

    The pointing directions of all the telescopes are set equally, but they can
    be modified individually (along with other porperties) later on.

    The telescope index is sorted first by y (from max to min) and then by x
    (from min to max), so that tel_index=0 corresponds to the telescope placed
    at the corner with minimum x and maximum y.

    Parameters
    ----------
    telescope : Telescope
        If None, the default GridElement object is used.
    x_c : float
        East coordinate in km of the center of the grid.
    y_c : float
        North coordinate in km of the center of the grid.
    z_c : float
        Height of the grid in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
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

    Attributes
    ----------
    x_c : float
        East coordinate in km of the center of the grid.
    y_c : float
        North coordinate in km of the center of the grid.
    z_c : float
        Height of the grid in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
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
    obs_type = 'Grid'

    def __init__(self, telescope=None, x_c=_x, y_c=_y, z_c=_z, theta=None,
                 alt=None, az=None, size_x=2., size_y=2., N_x=10, N_y=10):
        _grid(self, telescope, x_c, y_c, z_c, theta, alt, az, size_x,
              size_y, N_x, N_y)


# Constructors ################################################################
def _observatory(observatory):
    """
    Constructor of Observatory class.

    Parameters
    -----------
    observatory : Observatory
    """
    # observatory = Observatory([*telescopes])
    observatory.N_tel = len(observatory)
    observatory.size_x = None
    observatory.size_y = None

    # The first telescope is used as reference
    observatory.x_c = observatory[0].x
    observatory.y_c = observatory[0].y
    observatory.z_c = observatory[0].z
    observatory.tel_type = observatory[0].tel_type
    observatory.tel_apert = observatory[0].apert
    observatory.tel_area = observatory[0].area
    observatory.tel_N_pix = observatory[0].N_pix


def _array25(observatory, telescope, x_c, y_c, z_c, theta, alt, az, R,
             rot_angle):
    """
    Constructor of Array25 class.

    Parameters
    ----------
    observatory : Array25
    telescope : Telescope
        If None, the default IACT object is used.
    x_c : float
        East coordinate in km of the center of the array.
    y_c : float
        North coordinate in km of the center of the array.
    z_c : float
        Height of the array in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
    alt : float
        Altitude in degrees of the telescope pointing directions. If None,
        theta is used. If given, theta is overwritten.
    az : float
        Azimuth angle (from north, clockwise) in degrees of the telescope
        pointing directions.
    R : float
        Radius in km of the array.
    rot_angle : float
        Rotation angle in degrees of the array (clockwise).
    """
    if isinstance(telescope, Telescope):
        # Default theta and az values are taken from the input telescope
        if theta is None:
            theta = telescope.theta
        if az is None:
            az = telescope.az
        telescope = telescope.copy(x=x_c, y=y_c, z=z_c, theta=theta, alt=alt,
                                   az=az)
    elif telescope is None:
        # Default theta and az values are those of .telescope
        if theta is None:
            theta = _theta
        if az is None:
            az = _az
        # The default telescope type is IACT
        telescope = IACT(x_c, y_c, z_c, theta=theta, alt=alt, az=az)
    else:
        raise ValueError('The input telescope is not valid.')

    observatory.x_c = x_c
    observatory.y_c = y_c
    observatory.z_c = z_c
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    observatory.theta = theta
    observatory.alt = alt
    observatory.az = az
    observatory.size_x = 2.*R
    observatory.size_y = 2.*R
    observatory.R = R
    observatory.rot_angle = rot_angle
    observatory.tel_type = telescope.tel_type
    observatory.tel_apert = telescope.apert
    observatory.tel_area = telescope.area
    observatory.tel_Npix = telescope.N_pix

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


def _grid(grid, telescope, x_c, y_c, z_c, theta, alt, az, size_x,
          size_y, N_x, N_y):
    """
    Constructor of Grid class.

    Parameters
    ----------
    grid : Grid
    telescope : Telescope
        If None, the default GridElement object is used.
    x_c : float
        East coordinate in km of the center of the grid.
    y_c : float
        North coordinate in km of the center of the grid.
    z_c : float
        Height of the grid in km above ground level.
    theta : float
        Zenith angle in degrees of the telescope pointing directions.
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
    cell_area = step_x * step_y * 1000000.

    if isinstance(telescope, Telescope):
        # Default theta and az values are taken from the input telescope
        if theta is None:
            theta = telescope.theta
        if az is None:
            az = telescope.az
        telescope = telescope.copy(z=z_c, theta=theta, alt=alt, az=az)
    elif telescope is None:
        # The FoV is 180 degrees around zenith direction and area equal
        # to one grid cell
        theta = 0.
        alt = None
        az = 0.
        telescope = GridElement(z=z_c, theta=theta, alt=None, az=0.,
                                area=cell_area)
    else:
        raise ValueError('The input telescope is not valid.')

    grid.N_tel = N_x * N_y
    grid.x_c = x_c
    grid.y_c = y_c
    grid.z_c = z_c
    if alt is None:
        alt = 90. - theta
    else:
        theta = 90. - alt
    grid.theta = theta
    grid.alt = alt
    grid.az = az
    grid.size_x = size_x
    grid.size_y = size_y
    grid.N_x = N_x
    grid.N_y = N_y
    grid.cell_area = cell_area
    grid.tel_type = telescope.tel_type
    grid.tel_apert = telescope.apert
    grid.tel_area = telescope.area
    grid.tel_Npix = telescope.N_pix

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

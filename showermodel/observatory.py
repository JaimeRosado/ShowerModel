# coding: utf-8

import numpy as np
import showermodel as sm
import matplotlib.pyplot as plt

# Default values for array25
from .telescope import _x, _y, _z, _theta, _az


# Constructors ################################################################
def Observatory(*telescopes, obs_type=None):
    """
    Make an observatory from a list of telescopes.

    Parameters
    -----------
    *telescopes : list
        Arbitrary number of Telescope objects.
    obs_type : str or None
        Name assigned to the Observatory object, default None.

    Returns
    -------
    observatory : Observatory object.
    """
    from .telescope import _Telescope
    if not np.all([isinstance(tel, _Telescope) for tel in telescopes]):
        raise ValueError("Input telescopes are not valid.")
    observatory = _Observatory([*telescopes])
    observatory.obs_type = obs_type
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

    return observatory


def Array25(telescope=None, tel_type='IACT', x_c=_x, y_c=_y, z_c=_z,
            theta=None, alt=None, az=None, R=0.341, rot_angle=0.):
    """
    Make an array of 25 telescopes based on a layout of MST telescopes of CTA.

    The pointing directions of all the telescopes are set equally, but they can
    be modified individually (along with other porperties) later on.

    The telescope index is sorted first by radius and then by azimuth, so that
    tel_index=0 corresponds to the central telescope.

    Parameters
    ----------
    telescope : Telescope object
        If None, a new Telescope object of type tel_type is generated.
    tel_type : str
        Subclass of Telescope to be used when telescope==None. Default
        to IACT with field of view of 8 degrees, mirror area of 113 m^2 and
        1800 pixels. If tel_type==None, the parent class Telescope is used.
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

    Returns
    -------
    observatory : Observatory object.
    """
    from .telescope import _Telescope
    if isinstance(telescope, _Telescope):
        # Default theta and az values are taken from the input telescope
        if theta is None:
            theta = telescope.theta
        if az is None:
            az = telescope.az
        telescope = telescope.copy(x=x_c, y=y_c, z=z_c, theta=theta, alt=alt,
                                   az=az)  # tel_type is ignored
    elif telescope is None:
        # Default theta and az values are those of Telescope
        if theta is None:
            theta = _theta
        if az is None:
            az = _az
        telescope = sm.Telescope(x_c, y_c, z_c, theta=theta, alt=alt, az=az,
                                 tel_type=tel_type)
    else:
        raise ValueError('The input telescope is not valid.')

    observatory = _Array25()
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

    return observatory


def Grid(telescope=None, tel_type='GridElement', x_c=_x, y_c=_y, z_c=_z,
         theta=None, alt=None, az=None, size_x=2., size_y=2., N_x=10, N_y=10):
    """
    Make a rectangular grid of telescopes across the x and y directions.

    The pointing directions of all the telescopes are set equally, but they can
    be modified individually (along with other porperties) later on.

    The telescope index is sorted first by y (from max to min) and then by x
    (from min to max), so that tel_index=0 corresponds to the telescope placed
    at the corner with minimum x and maximum y.

    Parameters
    ----------
    telescope : Telescope object.
        If None, a new Telescope object of type tel_type is generated.
    tel_type : str
        Subclass of Telescope to be used when telescope==None. Default
        to GridElement with 100% detection efficiency, FoV of 180 degrees
        around zenith and area of one grid cell. If tel_type==None, the parent
        class Telescope is used.
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

    Returns
    -------
    observatory : Observatory object.
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

    from .telescope import _Telescope
    if isinstance(telescope, _Telescope):
        # Default theta and az values are taken from the input telescope
        if theta is None:
            theta = telescope.theta
        if az is None:
            az = telescope.az
        # tel_type is ignored
        telescope = telescope.copy(z=z_c, theta=theta, alt=alt, az=az)
    elif telescope is None:
        if tel_type == 'GridElement':
            # The FoV is 180 degrees around zenith direction and area equal
            # to one grid cell
            theta = 0.
            alt = None
            az = 0.
            telescope = sm.Telescope(z=z_c, theta=theta, alt=None, az=0.,
                                     tel_type=tel_type, area=cell_area)
        else:
            # Default theta and az values are those of Telescope
            if theta is None:
                theta = _theta
            if az is None:
                az = _az
            telescope = sm.Telescope(z=z_c, theta=theta, alt=None, az=az,
                                     tel_type=tel_type)
    else:
        raise ValueError('The input telescope is not valid.')

    grid = _Grid()
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

    return grid


# Classes #####################################################################
class _Observatory(list):
    """
    List containing the characterestics of an observatory and its constituent
    telescopes.

    Attributes
    ----------
    obs_type : str
        Name of the subclass of Observatory.
        Presently only the parent class Observatory and the Array25 and Grid
        subclasses are available. More subclasses to be implemented.
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
    R : float
        Radius of the observatory in km (only Array25 objects).
    rot_angle : float
        Rotation angle in degrees (clock-wise) of the array
        (only Array25 objects).
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

    Note: Attributes inherited from Telescope (i.e., tel_type, tel_apert,
    tel_area and tel_N_pix) are not updated when telescopes are modified
    or appended.

    Methods
    -------
    show : Show the telescope positions and indexes of the observatory in a
    2D plot.
    """
    obs_type = None

    def show(self):
        return _show(self)


class _Array25(_Observatory):
    obs_type = 'Array25'
    N_tel = 25
    pass


class _Grid(_Observatory):
    obs_type = 'Grid'
    pass


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

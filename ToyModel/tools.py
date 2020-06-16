# coding: utf-8

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings(
    'ignore',
    'Pandas doesn\'t allow columns to be created via a new attribute name',
    UserWarning)


def show_projection(projection, profile, shower_size, axes, max_theta, X_mark):
    """
    Show the projection of the shower track viewed by the telescope in both
    horizontal and FoV coordinates systems.
    """
    track = projection.track
    telescope = projection.telescope

    fig = plt.figure()
    # Plot in horizontal coordinates
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.set_title('Horizontal coordinates alt/az', y=1.1)

    # Plot in FoV coordinates
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_title('FoV coordinates theta/phi', y=1.1)

    # Only available for shower and event
    if shower_size:
        # For the shower track, the point radius is proportional to the shower
        # size
        shw_size = profile.N_ch
        shw_size = 50. * np.sqrt(shw_size / shw_size.max())
    else:
        shw_size = 20

    az = np.array(np.radians(projection.az))
    alt = np.array(projection.alt)
    phi = np.array(np.radians(projection.phi))
    theta = np.array(projection.theta)

    az_line = np.insert(az, 0, np.radians(projection.az_0))
    az_line = np.append(az_line, np.radians(projection.az_inf))
    alt_line = np.insert(alt, 0, projection.alt_0)
    alt_line = np.append(alt_line, projection.alt_inf)
    phi_line = np.insert(phi, 0, np.radians(projection.phi_0))
    phi_line = np.append(phi_line, np.radians(projection.phi_inf))
    theta_line = np.insert(theta, 0, projection.theta_0)
    theta_line = np.append(theta_line, projection.theta_inf)

    ax1.scatter(az, alt, c='r', s=shw_size, marker='o')
    ax1.plot(az_line, alt_line, 'r-')
    ax2.scatter(phi, theta, c='r', s=shw_size, marker='o')
    ax2.plot(phi, theta, 'r-')

    # Coordinates of the telescope FoV limits in FoV projection
    phi_FoV = np.linspace(0., 360., 61)
    theta_FoV = np.ones_like(phi_FoV) * telescope.apert / 2.
    # Horizontal coordinates system
    alt_FoV, az_FoV = telescope.thetaphi_to_altaz(theta_FoV, phi_FoV)

    ax1.plot(np.radians(az_FoV), alt_FoV, 'g')
    ax2.plot(np.radians(phi_FoV), theta_FoV, 'g')

    if axes:
        # Coordinates of the limits of a 2*pi solid angle around the telescope
        # pointing direction in FoV projection
        phi_2pi = phi_FoV  # = np.linspace(0., 360., 61)
        theta_2pi = np.ones(61) * 90.
        # Horizontal coordinates system
        alt_2pi, az_2pi = telescope.thetaphi_to_altaz(theta_2pi, phi_2pi)
        ax1.plot(np.radians(az_2pi), alt_2pi, 'g--')

        # Coordinates of the horizon in horizontal projection
        az_hor = phi_FoV  # = np.linspace(0., 360., 61)
        alt_hor = np.zeros(61)
        # In FoV projection
        theta_hor, phi_hor = telescope.altaz_to_thetaphi(alt_hor, az_hor)
        ax2.plot(np.radians(phi_hor), theta_hor, 'g--')

        # Coordinates of west-zenith-east arc in horizontal coordinates system
        # Values greater than 90 degrees correspond to az_we = 180 degrees
        alt_we = np.linspace(0., 180., 61)
        az_we = alt_hor  # np.zeros(61)
        # In FoV projection
        theta_we, phi_we = telescope.altaz_to_thetaphi(alt_we, az_we)
        ax2.plot(np.radians(phi_we), theta_we, 'g--')

        # Coordinates of south-zenith-north arc in horiziontal coordinates
        # system
        alt_sn = alt_we  # np.linspace(0., 180., 61)
        az_sn = theta_2pi  # =np.ones(61) * 90.
        # In FoV projection
        theta_sn, phi_sn = telescope.altaz_to_thetaphi(alt_sn, az_sn)
        ax2.plot(np.radians(phi_sn), theta_sn, 'g--')

        # Coordinates of the arc phi=0/180 (going through north point and the
        # telescope pointing direction) in FoV coordinates system
        theta_phi0 = alt_we - 90.  # = np.linspace(-90., 90., 61)
        # Negative values correspond to phi = 180 degrees
        phi_phi0 = az_we  # =np.zeros(61)
        # Horizontal coordinates system
        alt_phi0, az_phi0 = telescope.thetaphi_to_altaz(theta_phi0, phi_phi0)
        ax1.plot(np.radians(az_phi0), alt_phi0, 'g--')

        # Coordinates of the arc phi=90/270 in FoV coordinates system
        theta_phi90 = theta_phi0  # = np.linspace(-90., 90., 61)
        # Negative values correspond to phi = 270 degrees
        phi_phi90 = theta_2pi  # =np.ones(61) * 90.
        # Horizontal coordinates system
        alt_phi90, az_phi90 = telescope.thetaphi_to_altaz(theta_phi90,
                                                          phi_phi90)
        ax1.plot(np.radians(az_phi90), alt_phi90, 'g--')

    if X_mark is not None:
        # Absolute coordinates of the shower point at the input
        # slanth depth X_mark
        # x_mark, y_mark, z_mark = telescope._abs_to_rel(
        #              *track.X_to_xyz(X_mark))
        x_mark, y_mark, z_mark = track.X_to_xyz(X_mark)
        r_mark, alt_mark, az_mark, theta_mark, phi_mark = telescope.spherical(
            x_mark, y_mark, z_mark)

        ax1.plot(np.radians(az_mark), alt_mark, 'bx')
        ax2.plot(np.radians(phi_mark), theta_mark, 'bx')

    ax1.set_rmin(90.)
    ax1.set_rmax(0.)
    # Polar angle labels in the plot refers to azimuth
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(225)

    ax2.set_rmin(0.)
    ax2.set_rmax(max_theta)
    ax2.set_rlabel_position(-45)
    # Offset should be the north direction from right-hand direction
    ax2.set_theta_zero_location('E', offset=telescope.phi_right)
    ax2.set_theta_direction(-1)
    # theta is the name of the axial angle in the plot, not the
    # coordinate theta !
    plt.tight_layout()

    return ax1, ax2


def show_geometry(obj, observatory, mode, x_min, x_max, y_min, y_max,
                  X_mark, shower_size, signal_size, tel_index, xy_proj,
                  pointing):
    """
    Show a shower track together with the telescope positions of an observatory
    in an either 2D or 3D plot.
    """
    from .telescope import _Telescope
    from .observatory import _Observatory
    if not isinstance(observatory, (_Telescope, _Observatory)):
        raise ValueError('The input observatory is not valid')
    elif isinstance(observatory, _Telescope):
        telescope = observatory
        observatory = _Observatory()
        observatory.append(telescope)

    from .track import _Track
    track = obj.track if not isinstance(obj, _Track) else obj

    # Track points
    data_range = ((x_min < track.x) & (track.x < x_max) & (y_min < track.y)
                  & (track.y < y_max))
    x_track = np.array(track.x[data_range])
    y_track = np.array(track.y[data_range])
    z_track = np.array(track.z[data_range])

    # Telescope positions
    coords = [(telescope.x, telescope.y, telescope.z) for telescope
              in observatory]
    x_tel, y_tel, z_tel = zip(*coords)

    z_tel_min = min(z_tel)  # z coordinate of the most lower telescope
    z_tel_max = max(z_tel)  # z coordinate of the most upper telescope

    # Track line reaching observation level
    x_line = x_track.copy()
    y_line = y_track.copy()
    z_line = z_track.copy()
    # The observation level is lower than the lowest track point
    if z_tel_min < z_track.min():
        x_line = np.insert(x_line, 0,
                           track.x0 + z_tel_min * track.ux / track.uz)
        y_line = np.insert(y_line, 0,
                           track.y0 + z_tel_min * track.uy / track.uz)
        z_line = np.insert(z_line, 0, z_tel_min)
    # The observation level is higher than the highest track point
    if z_tel_max > z_track.max():
        x_line = np.append(x_line, track.x0 + z_tel_max * track.ux / track.uz)
        y_line = np.append(y_line, track.y0 + z_tel_max * track.uy / track.uz)
        z_line = np.append(z_line, z_tel_max)
    z_min = z_line.min()
    z_max = z_line.max()

    if shower_size:  # Only available for shower and event
        # For the shower track, the point radius is proportional to the shower
        # size
        shw_size = obj.profile.N_ch[data_range]
        shw_size = 50. * np.sqrt(shw_size / shw_size.max())
    else:
        shw_size = 20

    if signal_size:  # Only available for event, when some signal is produced
        # For the telescope positions, the point radius is prportional to
        # the signal
        signal_size = np.array([signal.Npe_total_sum for signal
                                in obj.signals])
        signal_size_max = signal_size.max()
        if signal_size_max > 0.:
            signal_size = 50. * np.sqrt(signal_size / signal_size.max())
        else:
            signal_size = 20
    else:
        signal_size = 20

    if mode == '2d':  # 2d projection
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)
        # Telescope positions in green
        ax.scatter(x_tel, y_tel, c='g', s=signal_size, marker='o')
        # Shower track line in red
        ax.plot(x_line, y_line, 'r-')
        # Shower track points in red
        ax.scatter(x_track, y_track, c='r', s=shw_size, marker='o')

        if tel_index:
            for tel_index, telescope in enumerate(observatory):
                ax.annotate(tel_index, (x_tel[tel_index], y_tel[tel_index]))

        if X_mark is not None:
            # Coordinates of the shower corresponding to the input slant
            # depth X_mark
            x_mark, y_mark, z_mark = track.X_to_xyz(X_mark)
            ax.scatter(x_mark, y_mark, c='b', marker='x')

    else:  # 3d projection
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')
        # Telescope positions in green
        ax.scatter(x_tel, y_tel, z_tel, c='g', s=signal_size, marker='o')
        ax.plot(x_line, y_line, z_line, 'r-')  # Shower track line in red
        if xy_proj:
            # xy projection of shower track line in red
            ax.plot(x_line, y_line, 'r--', zs=z_min, zdir='z')
            # Shower track points in red
        ax.scatter(x_track, y_track, z_track, c='r', s=shw_size, marker='o')

        if pointing:
            # Arrows pointing to the telescope axis direction
            pointing = [(telescope.ux, telescope.uy, telescope.uz)
                        for telescope in observatory]
            ux, uy, uz = zip(*pointing)
            ax.quiver(x_tel, y_tel, z_tel, ux, uy, uz, length=0.4)

        if X_mark is not None:
            # Coordinates of the shower corresponding to the input slanth
            # depth X_mark
            x_mark, y_mark, z_mark = track.X_to_xyz(X_mark)
            if (z_min < z_mark) & (z_mark < z_max):
                ax.scatter(x_mark, y_mark, z_mark, c='b', marker='x')

        # Auto-scale for z
        ax.set_zlim(z_min, z_max)
        ax.axes.zaxis.set_label_text('z (km)')

    ax.axis([x_min, x_max, y_min, y_max])
    ax.axes.xaxis.set_label_text('x (km)')
    ax.axes.yaxis.set_label_text('y (km)')
    return ax


def rotate(vx, vy, vz, rot_angle, x, y, z):
    """
    Rotate the vector (x,y,z) by an angle rot_angle (positive or negative)
    around the axis (vx,vy,vz), where v is a unit vector.
    """
    ct = math.cos(math.radians(rot_angle))
    st = math.sin(math.radians(rot_angle))
    x_rot = ((ct+vx*vx*(1.-ct)) * x + (vx*vy*(1.-ct)-vz*st) * y
             + (vx*vz*(1.-ct)+vy*st) * z)
    y_rot = ((vx*vy*(1.-ct)+vz*st) * x + (ct+vy*vy*(1.-ct)) * y
             + (vy*vz*(1.-ct)-vx*st) * z)
    z_rot = ((vx*vz*(1.-ct)-vy*st) * x + (vy*vz*(1.-ct)+vx*st) * y
             + (ct+vz*vz*(1.-ct)) * z)

    return (x_rot, y_rot, z_rot)


def zr_to_theta(z, r):
    """
    Calculate the angle theta in degrees [0, 180] of a vector with vertical
    projection z and modulus r, where theta is defined from the z axis.
    """
    try:  # For r being a scalar (otherwise r==0 generates an exception)
        if r == 0:
            theta = 90. * np.ones_like(z)
        else:
            z_r = np.array(z/r)
            z_r[z_r > 1.] = 1.
            theta = np.degrees(np.arccos(z_r))
        # If z is also a scalar, the function returns a scalar.
        # Otherwise, the output is an array
        return 1.*theta

    except Exception:  # For r being an array (or a Series)
        r = np.array(r)
        # For z being a scalar
        if not isinstance(z, (tuple, list, np.ndarray, pd.Series)):
            # An array of same lenght as r is generated
            z = np.ones_like(r) * z
        else:
            z = np.array(z)
        z_r = np.zeros_like(r)
        z_r[r != 0] = z[r != 0] / r[r != 0]
        z_r[z_r > 1.] = 1.

        theta = 90. * np.ones_like(r)
        theta[r != 0] = np.degrees(np.arccos(z_r[r != 0]))
        return theta


def xy_to_phi(x, y):
    """
    Calculate the angle phi in degrees [-90, 270) of the xy projection of a
    vector, where phi is defined from the x axis towards the y axis
    (anticlockwise).
    """
    try:  # For x being a scalar (otherwise x==0 generates an exception)
        if x == 0:
            phi = np.sign(y) * 90.
        else:
            phi = np.degrees(np.arctan(y/x))
            if x < 0:
                phi = phi + 180.
        # If y is also a scalar, the function returns a scalar.
        # Otherwise, the output is an array
        return 1.*phi

    except Exception:  # For x being an array (or a Series)
        x = np.array(x)
        # For r being a scalar
        if not isinstance(y, (tuple, list, np.ndarray, pd.Series)):
            # An array of same lenght as x is generated
            y = np.ones_like(x) * y
        else:
            y = np.array(y)

        phi = np.zeros_like(x)
        phi[x == 0] = np.sign(y[x == 0]) * 90.
        phi[x != 0] = np.degrees(np.arctan(y[x != 0] / x[x != 0]))
        phi[x < 0] = phi[x < 0] + 180.
        return phi

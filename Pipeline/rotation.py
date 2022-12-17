# useful functions for coordinate transformation

import numpy as np
import scipy.constants as const
import healpy as hp


ra_G = 192.85948*const.degree     # RA of the galactic center [radian]
dec_G = 27.12825*const.degree     # DEC of the galactic center [radian]


# a function that rotates galactic coordinate to equatorial coordinate
r_G2C = hp.rotator.Rotator(coord = ('G', 'C'))



def rot_cyl2cart(theta):
    """
    Returns the rotation matrix from cylindrical to cartesian coordinate
    system.

    Parameters
    ----------
    theta : float
        the theta angle in a cylindrcal coordinate system [degree]

    Returns
    -------
    The rotation matrix from cylindrical to cartesian coordinate system given
    the angle theta.
    """
    theta = theta*const.degree
    r_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta),  np.cos(theta), 0],
                      [            0,              0, 1]])
    return r_mat


def rot_cart2galactic(l, b):
    """
    Returns the rotation matrix from cartesian coordinate system to galactic
    coordinate system.

    Parameters
    ----------
    l : float
        galactic longtitude [degree]
    b : float
        galactic lattitude [degree]

    Returns
    -------
    The rotation matrix from cartesian coordinate system to galactic coordinate
    system given galactic coordinate (l, b)
    """
    l = l*const.degree
    b = b*const.degree
    r_mat = np.array([[ np.cos(b)*np.cos(l),  np.cos(b)*np.sin(l), np.sin(b)],
                      [          -np.sin(l),            np.cos(l),         0],
                      [-np.sin(b)*np.cos(l), -np.sin(b)*np.sin(l), np.cos(b)]])
    return r_mat


def rot_vec_radec2lb(l, b):
    """
    Returns the rotation matrix for rotating a 2D vector in the basis of (l , b)
    given in the basis (RA, DEC).

    Parameters
    ----------
    vra : float
        vector component in the RA direction
    vdec : float
        vector component in the DEC direction
    l : float
        galactic longtitude [degree]
    b : float
        galactic lattitude [degree]

    Returns
    -------
    The 2D rotation matrix
    """

    l = l*const.degree
    b = b*const.degree

    dec, ra = r_G2C(np.pi/2 - b, l)
    dec = np.pi/2 - dec
    cos = (np.sin(dec_G) - np.sin(dec)*np.sin(b))/np.cos(dec)/np.cos(b)
    sin = np.sin(ra - ra_G)*np.cos(dec_G)/np.cos(b)
    r_matrix = np.array([[ cos, sin],
                         [-sin, cos]])

    return r_matrix

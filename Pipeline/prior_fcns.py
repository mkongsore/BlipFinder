import numpy as np
import scipy.constants as const



class Priors():
    """
    The parent class for the priors of various lensing objects.

    Attributes
    ----------
    r0 : float
        sun's distance to the galactic center in [kpc]
    sun_rot : np.array
        sun's rotational velocity in the conventional UVW cartesian coordinate [km/s]
    sun_pec : np.array
        sun's peculiar velocity in the conventional UVW cartesian coordinate [km/s]
    """


    def __init__(self, r0 = 8., sun_rot = np.array([0., 220., 0.]),
                 sun_pec = np.array([11.1, 12.24, 7.25])):
        """
        Class constructor. Set the information of the sun

        Parameters
        ----------
        r0 : float
            sun's distance to the galactic center in [kpc] (default 8.0)
        sun_rot : np.array
            sun's rotational velocity in the conventional UVW cartesian coordinate [km/s]
            (default [0., 220., 0.])
        sun_pec : np.array
            sun's peculiar velocity in the conventional UVW cartesian coordinate [km/s]
            (default [11.1, 12.24, 7.25])
        """
        self.r0 = r0
        self.sun_rot = sun_rot
        self.sun_pec = sun_pec

    ##### helper methods #####

    def _rho_cyl(self, r, l, b):
        """
        Returns the distance to the galactic center in kpc given the
        distance to the solar system and galactic coordinates in cylindrical
        coordinate.

        Parameters
        ----------
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]

        Returns
        -------
        The distance to galactic center given galactic coordinate (r, l, b) [kpc]
        """
        l = l*const.degree
        b = b*const.degree
        r0 = self.r0
        return np.sqrt(r0**2 + (r*np.cos(b))**2 - 2*r*r0*np.cos(l)*np.cos(b))


    def _theta_cyl(self, r, l, b):
        """
        Returns the theta angle in a galacto-centric cylindrical coordinate
        system.

        Parameters
        ----------
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]

        Returns
        -------
        The theta angle in a galacto-centric cylindrical coordinate system
        given galactic coordinate (r, l, b) [kpc]
        """
        r0 = self.r0
        rho = self._rho_cyl(r, l, b)
        r = r*np.cos(b*const.degree)
        cos = (rho**2 + r0**2 - r**2)/(2*rho*r0)
        if cos > 1:
            cos = 1
        theta = np.arccos(cos)/const.degree
        theta = 180 - theta
        if l > 180:
            theta = 360 - theta
        return theta


    def _z_cyl(self, r, b):
        """
        Returns the vertical distance to the galactic disk in kpc given the
        distance to the solar system and galactic coordinates in cylinrical
        coordinate.

        Parameters
        ----------
        r : np.array
            distance to the solar system [kpc]
        b : float
            galactic lattitude [degree]

        Returns
        -------
        The vertical distance to the galactic disk given galactic coordinate
        (r, b) [kpc]
        """
        b = b*const.degree
        return r*np.sin(b)


    def _rho_sph(self, r, l, b):
        """
        Returns the distance to the galactic center in kpc given the
        distance to the solar system and galactic coordinates in spherical
        coordinate.

        Parameters
        ----------
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]

        Returns
        -------
        The distance to galactic center given galactic coordinate (r, l, b) [kpc]
        """
        l = l*const.degree
        b = b*const.degree
        r0 = self.r0
        return np.sqrt(r0**2 + r**2 - 2*r*r0*np.cos(l)*np.cos(b))


    def _disk_model(self, r, l, b, rd, zd):
        """
        Returns the (un-normalized) pdf of a disk distribution given galactic coordinate

        Parameters
        ----------
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]
        rd : float
            scale radius of the disk model [kpc]
        zd : float
            scale height of the disk model [kpc]

        Returns
        -------
        The un-normalized pdf of a disk distribution given galactic coordinate (r, l, b)
        """
        rho = self._rho_cyl(r, l, b)
        z = self._z_cyl(r, b)
        return np.exp(-np.abs(z)/zd - rho/rd)


    def _NFW_model(self, r, l, b, rs):
        """
        Returns the (un-normalized) pdf of a disk distribution given galactic coordinate

        Parameters
        ----------
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]
        rs : float
            scale radius of the NFW model [kpc]

        Returns
        -------
        The un-normalized pdf of a NFW distribution given galactic coordinate (r, l, b)
        """
        rho = self._rho_sph(r, l, b)
        x = rho/rs
        return 1/(x*(1 + x)**2)

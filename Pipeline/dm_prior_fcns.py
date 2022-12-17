import numpy as np
import scipy.constants as const
import rotation as rot
import prior_fcns as prior



class DM_priors(prior.Priors):
    """
    The class of the priors of dense dark matter substructures. 
    An inherent class of Priors.

    Attributes
    ----------
    rs : float
        scale radius of the Milky Way halo NFW model [kpc] 
    rho_s : float
        scale density of the Milky Way halo NFW model [M_sun/pc^3]
    s_v : float
        velocity dispersion [km/s]
    """


    def __init__(self, r0 = 8,
                 sun_rot = np.array([0, 220, 0]),
                 sun_pec = np.array([11.1, 12.24, 7.25]),
                 rs = 18., rho_s = 3e-3,
                 s_v = 166.):
        """
        Class constructor. Set the information of the BH

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
        rs : float
            scale radius of the Milky Way halo NFW model [kpc] (default 18)
        rho_s : float
            scale density of the Milky Way halo NFW model [M_sun/pc^3] (default 3e-3)
        s_v : float
            velocity dispersion [km/s] (default 166.)
        """
        super().__init__(r0, sun_rot, sun_pec)
        self.rs = rs
        self.rho_s = rho_s
        self.s_v = s_v


    def joint_pdf_DM(self, pmra, pmdec, r, l, b):
        """
        Returns the joint pdf P(pm_ra, pm_dec, r)

        Parameters
        ----------
        pmra : float
            proper motion in the RA direction [mas/yr]
        pmdec : float
            proper motion in the DEC direction [mas/yr]
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]

        Returns
        -------
        The un-normalized pdf of a BH with proper motion (pmra, pmdec) at given
        galactic coordinate (r, l, b) in the solar rest frame
        """
        # Conditional probability P(pmra, pmdec | r)
        pm = self._prior_radec(pmra, pmdec, r, l, b)
        # marginalized probability P(r)
        dist = self._pdf_DM_distance(r, l, b)

        # Bayes Theorem: P(pmra, pmdec, r) = P(pmra, pmdec | r) * P(r)
        return pm*dist


    ##### helper methods #####

    def _prior_radec(self, pmra, pmdec, r, l, b):
        """
        Returns the normalized conditional probability P(pm_ra, pm_dec | d)

        Parameters
        ----------
        pmra : float
            proper motion in the RA direction [mas/yr]
        pmdec : float
            proper motion in the DEC direction [mas/yr]
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]

        Returns
        -------
        The normalized conditional pdf P(pmra, pmdec | r) of a BH at given
        galactic coordinate (r, l, b) in the solar rest frame
        """
        # Jacobian factor for changing variable from proper motion [mas/yr]
        # to velocity [km/s]
        jac = r*const.parsec/const.Julian_year*const.arcsec/1000

        # rotate from (pmra, pmdec) to (pml, pmb)
        pm = rot.rot_vec_radec2lb(l, b).dot([pmra, pmdec])*jac

        vl = pm[0]
        vb = pm[1]

        return self._prior_lb(vl, vb, r, l, b)*jac**2


    def _prior_lb(self, vl, vb, r, l, b):
        """
        Returns the normalized conditional probability P(v_l, v_b|r)

        Parameters
        ----------
        vl : float
            velocity in the l direction [km/s]
        vb : float
            velocity in the b direction [km/s]
        r : np.array
            distance to the solar system [kpc]
        l : float
            galactic longtitude [degree]
        b : float
            galactic lattitude [degree]

        Returns
        -------
        The normalized conditional pdf P(vl, vb | r) of a BH at given
        galactic coordinate (r, l, b) in the solar rest frame
        """

        theta = self._theta_cyl(r, l, b)
        r2_mat = rot.rot_cart2galactic(l, b)

        v = np.array([vl, vb])
        v_sun = r2_mat.dot(self.sun_rot + self.sun_pec)

        # peculiar velocity of the BH in its local frame of rest
        v = v + v_sun[1:]


        chi_sq = v.dot(v)/(2*self.s_v**2)

        return 1/(2*np.pi)/self.s_v**2*np.exp(-chi_sq/2)


    def _pdf_DM_distance(self, r, l, b):
        """
        Returns the (non-normalized) PDF of BH distance away from the
        solar system.

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
        The un-normalized pdf P(r) of a BH at given galactic coordinate (r, l, b)
        """
        return r**2*self._NFW_model(r, l, b, self.rs)

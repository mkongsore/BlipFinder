import numpy as np
import scipy.constants as const
import rotation as rot
import prior_fcns as prior



class BH_priors(prior.Priors):
    """
    The class of the priors of astrophysical black holes. An inherent class of Priors

    Attributes
    ----------
    v_rot : np.array
        BH's rotational velocity in cylindrical coordinate [km/s]
    rd : float
        scale radius of the disk model [kpc]
    zd : float
        scale height of the disk model [kpc]
    sr : float
        velocity dispersion in the r diection [km/s]
    sphi : float
        velocity dispersion in the phi diection [km/s]
    sz : float
        velocity dispersion in the z diection [km/s]
    """


    def __init__(self, r0 = 8,
                 sun_rot = np.array([0, 220, 0]),
                 sun_pec = np.array([11.1, 12.24, 7.25]),
                 v_rot = np.array([0, -220, 0]),
                 rd = 2.6, zd = 3.,
                 sr = 77.5, sphi = 72.5, sz = 70.):
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
        v_rot : np.array
            BH's rotational velocity in cylindrical coordinate [km/s]
            (default [0., -220., 0.])
        rd : float
            scale radius of the disk model [kpc] (default 2.6)
        zd : float
            scale height of the disk model [kpc] (default 3.0)
        sr : float
            velocity dispersion in the r diection [km/s] (default 77.5)
        sphi : float
            velocity dispersion in the phi diection [km/s] (default 72.5)
        sz : float
            velocity dispersion in the z diection [km/s] (default 70.0)
        """
        super().__init__(r0, sun_rot, sun_pec)
        self.v_rot = v_rot
        self.rd = rd
        self.zd = zd
        self.sr = sr
        self.sphi = sphi
        self.sz = sz


    def joint_pdf_BH(self, pmra, pmdec, r, l, b):
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
        dist = self._pdf_BH_distance(r, l, b)

        # Bayes Theorem: P(pmra, pmdec, r) = P(pmra, pmdec | r) * P(r)
        return pm*dist


    def mass_BH(self, mass, a = 2.63, m_min = 4.59, m_max = 86.22, frac = 0.1,
             dm = 4.82, mu = 33.07, sigma = 5.69):
        """
        Returns the normalized BH mass distribution given the best fit parameters
        of the LIGO posterior of the POWER LAW + PEAK model
        (See arxiv.org/abs/2010.14533 Fig.16)

        Prameters
        ---------
        mass : float
            mass of the BH [M_sun]
        a : float
            slope of the power law of BH mass distribution
        m_min : flaot
            lower bound of the BH mass [M_sun]
        m_max : flaot
            upper bound of the BH mass [M_sun]
        frac : float
            fraction of the gaussian peak from the POWER LAW + PEAK model
        dm : float
            range of the smoothing around the lower bound [M_sun]
        mu : float
            average of the gaussian peak [M_sun]
        sigma : float
            standard deviation of the gaussion peak [M_sun]

        Returns
        -------
        The normalized BH mass distribution
        """

        return ((1 - frac)*_power_law(mass, a, m_min, m_max) +
                frac*_gaussian(mass, mu, sigma))*_smooth(mass, m_min, dm)


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
        sr = self.sr
        sphi = self.sphi
        sz = self.sz

        # diagonal covariance matrix
        diag_cov = np.diag([1/sr**2, 1/sphi**2, 1/sz**2])

        theta = self._theta_cyl(r, l, b)
        r1_mat = rot.rot_cyl2cart(theta)
        r2_mat = rot.rot_cart2galactic(l, b)

        v = np.array([vl, vb])
        v0 = r2_mat.dot(r1_mat.dot(self.v_rot))
        v_sun = r2_mat.dot(self.sun_rot + self.sun_pec)

        # peculiar velocity of the BH in its local frame of rest
        v = v - (v0 - v_sun)[1:]

        inv_cov = r1_mat.dot(diag_cov.dot(r1_mat.T))
        inv_cov = r2_mat.dot(inv_cov.dot(r2_mat.T))

        # the marginalized covariance matrix after integrating the radial direction
        reduced_cov = np.zeros((2,2))
        reduced_cov[0,0] = inv_cov[1,1] - inv_cov[0,1]**2/inv_cov[0,0]
        reduced_cov[1,1] = inv_cov[2,2] - inv_cov[0,2]**2/inv_cov[0,0]
        reduced_cov[0,1] = inv_cov[1,2] - inv_cov[0,1]*inv_cov[0,2]/inv_cov[0,0]
        reduced_cov[1,0] = reduced_cov[0,1]

        chi_sq = v.dot(reduced_cov.dot(v))

        return 1/(2*np.pi)/(sr*sphi*sz)*np.exp(-chi_sq/2)/np.sqrt(inv_cov[0,0])


    def _pdf_BH_distance(self, r, l, b):
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
        return r**2*self._disk_model(r, l, b, self.rd, self.zd)


##### helper function for BH mass pdf #####

def _power_law(x, a, m_min, m_max):
    """
    Returns the normalized power law distribution with lower and upper bounds

    Parameters
    ----------
    x : float
        input fot the distribution
    a : float
        slope of the power law
    m_min : flaot
        lower bound
    m_max : flaot
        upper bound

    Returns
    -------
    The normalized power law distribution with lower and upper bound
    """
    return np.heaviside(m_max - x, 1)*np.heaviside(x - m_min, 1)*x**(-a)/(m_max**(1-a) - m_min**(1-a))*(1-a)


def _gaussian(x, mu, sigma):
    """
    Returns the normalized gaussian distribution

    Parameters
    ----------
    x : float
        input fot the distribution
    mu : float
        average of the gaussian distribution
    sigma : flaot
        standard deviation of the gaussian distribution

    Returns
    -------
    The normalized normalized gaussian distribution
    """
    return np.exp(-(x - mu)**2/(2*sigma**2))/np.sqrt(2*np.pi)/sigma


def _smooth(x, m_min, dm):
    """
    Returns the smoothing function around the lower bound

    Parameters
    ----------
    x : float/np.array
        input fot the distribution
    m_min : float
        lower bound
    dm : flaot
        range of the smoothing process

    Returns
    -------
    The smoothing function around the lower bound
    """
    # if x is np.array
    try:
        new_x = x[:]
        new_x[new_x == m_min] = m_min + 1e-3
        new_x[new_x == m_min + dm] = m_min + dm + 1e-3
    # if x is float
    except:
        if x == m_min:
            new_x = m_min + 1e-3
        elif x == m_min + dm:
            new_x = m_min + dm + 1e-3
        else:
            new_x = x

    f = np.exp(dm/(new_x - m_min) + dm/(new_x - m_min - dm))

    return np.heaviside(x - m_min, 0)*np.heaviside(m_min + dm - x, 0)/(f + 1) + np.heaviside(x - m_min - dm, 1)

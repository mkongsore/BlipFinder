# A class of functions for generating stellar motion and lesning events as seen in GAIA.

import numpy as np
import pandas as pd
import scipy.constants as const
import scipy.integrate as integrate
import scipy.special as sp
import scipy.optimize as op

_M_sun = 1.98847e30 # solar mass in kg

class Dynamics():
    """
    A class containing all functions for generating mock stellar motion and lensing event data.

    Attributes
    ----------
    obs_info : pandas.DataFrame
        table of observation information.
    t_obs : pandas.Series
        series of the observation time in [Julian year]
    scan_angle : pandas.Series
        series of the scan angle of GAIA in [radians]
    t_ref : float
        reference time for the referenced location (RA, DEC) in the sky [Julian year]
    n_obs : int
        number of observations
    earth_loc : np.array
        Cartesian coordinate of the earth at t_obs in [AU]
    """

    ##### Class invariants: some constants for parallax calculations #####
    _ecc =  0.01671                      # eeccentricity of the earth
    _t_peri = 0.006859071868393585       # averaged time of perihelion from 2015 - 2020 in Julian year
    _obliq = 23.44*const.degree          # obliquity of the earth in radians
    _t_eq = 0.21542231434473252 - _t_peri # time of spring equinox counting from t_peri
    _m_eq = 2*np.pi*_t_eq                 # mean anomaly at t_eq
    _E_eq = _m_eq + _ecc*np.sin(_m_eq)      # eccentric anomaly at t_eq
    _theta_eq = 2*np.arctan(np.tan(_E_eq/2)*np.sqrt((1+_ecc)/(1-_ecc))) # polar angle theta at t_eq
                                                                        # with 0 at spring equinox
    _obliq_rotation = np.array([[1,             0,              0],
                                [0, np.cos(_obliq), -np.sin(_obliq)],
                                [0, np.sin(_obliq),  np.cos(_obliq)]])



    def __init__(self, obs_info, t_ref = 2017.5):

        """
        Class constructor. Sets basic astrophysical parameters for constructing events.

        Parameters
        ----------
        obs_info : str
            File name that stores the observation information table.
            table should have two columns, first column is the observation time stamps
            in Julian year, second column is the corresponding scan angle in radians.
            (default: 'obs_info.csv')
        t_ref : float/int
            reference time for the referenced location (RA, DEC) in the sky [Julian year]
            (default: 2017.5)
        """

        # Load file containing time and scan angle information for all observations
        self.obs_info = pd.read_csv(obs_info, sep=",", header=None)

        self.t_obs = (self.obs_info).loc[0].to_numpy()
        self.t_plot = np.linspace(self.t_obs[0],self.t_obs[-1],100000) # Time steps for plotting [Julian years]
        self.scan_angles = (self.obs_info).loc[1].to_numpy() #  Scan angles [rad]
        self.t_ref = t_ref # Reference time for calculating displacement [Julian years]
        self.n_obs = len(self.t_obs) # number of observations
        self.earth_loc = self._earth_cart() # earth's location at time t_obs


    def unlens_AL(self, ra0, dec0, pmra, pmdec, dist):
        """
        A method for computing the AL displacement of an object in the sky
        at all observation times given its 5 parameter astrometric solutions.

        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]
        pmra : float
            proper motion in the RA* direction of the object [mas/yr]
        pmdec : float
            proper motion in the DEC direction of the object [mas/yr]
        dist : float
            distance to the object [pc]

        Returns
        -------
        Array of along scan displacements in mas
        """

        ra_ar, dec_ar = self._trajectory(ra0, dec0, pmra, pmdec, dist)

        return self._al_disp(ra_ar, dec_ar)


    def unlens7p_AL(self, ra0, dec0, pmra, pmdec, dist, accra, accdec):
        """
        A method for computing the AL displacement of an object in the sky
        at all observation times given its 5 parameter astrometric solutions.

        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]
        pmra : float
            proper motion in the RA* direction of the object [mas/yr]
        pmdec : float
            proper motion in the DEC direction of the object [mas/yr]
        dist : float
            distance to the object [pc]

        Returns
        -------
        Array of along scan displacements in mas
        """

        ra_ar, dec_ar = self._trajectory_7p(ra0, dec0, pmra, pmdec, dist, accra, accdec)

        return self._al_disp(ra_ar, dec_ar)



    def unlens_2d(self, ra0, dec0, pmra, pmdec, dist):
        """
        A method for computing the displacement of an object in the sky in equatorial coordinates
        at all observation times given its 5 parameter astrometric solutions.

        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]
        pmra : float
            proper motion in the RA* direction of the object [mas/yr]
        pmdec : float
            proper motion in the DEC direction of the object [mas/yr]
        dist : float
            distance to the object [pc]

        Returns
        -------
        Double of displacement of object in equatorial coordinates with unit mas
        """

        ra_ar, dec_ar = self._trajectory(ra0, dec0, pmra, pmdec, dist)

        return (ra_ar, dec_ar)


    def lensed_AL(self, ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                        ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass):
        """
        A method for computing the AL displacement of an object in the sky
        at all observation times given its 11 parameter lensed astrometric solutions.

        Parameters
        ----------
        ra_s : float
            reference RA of the source [degrees]
        dec_s : float
            reference DEC of the source [degrees]
        pmra_s : float
            proper motion in the RA* direction of the source [mas/yr]
        pmdec_s : float
            proper motion in the DEC direction of the source [mas/yr]
        dist_s : float
            distance to the source [pc]
        ra_l : float
            reference RA of the lens [degrees]
        dec_l : float
            reference DEC of the lens [degrees]
        pmra_l : float
            proper motion in the RA* direction of the lens [mas/yr]
        pmdec_l : float
            proper motion in the DEC direction of the lens [mas/yr]
        dist_l : float
            distance to the lens [pc]
        mass : float
            mass of the lens

        Returns
        -------
        Array of along scan displacements in mas
        """

        ra_ar, dec_ar = self._trajectory_lensed(ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                                     ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass)

        return self._al_disp(ra_ar, dec_ar)


    def lensed_AL_mag(self, ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                        ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass):
        """
        A method for computing the AL displacement of an object in the sky
        at all observation times given its 11 parameter lensed astrometric solutions.

        Parameters
        ----------
        ra_s : float
            reference RA of the source [degrees]
        dec_s : float
            reference DEC of the source [degrees]
        pmra_s : float
            proper motion in the RA* direction of the source [mas/yr]
        pmdec_s : float
            proper motion in the DEC direction of the source [mas/yr]
        dist_s : float
            distance to the source [pc]
        ra_l : float
            reference RA of the lens [degrees]
        dec_l : float
            reference DEC of the lens [degrees]
        pmra_l : float
            proper motion in the RA* direction of the lens [mas/yr]
        pmdec_l : float
            proper motion in the DEC direction of the lens [mas/yr]
        dist_l : float
            distance to the lens [pc]
        mass : float
            mass of the lens

        Returns
        -------
        Array of along scan displacements in mas
        """

        ra_ar, dec_ar, mag = self._traj_lensed_mag(ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                                     ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass)

        return self._al_disp(ra_ar, dec_ar), mag


    def binary_AL(self, ra0, dec0, pmra, pmdec, dist, r_sep, mass, m_dark, ecc, theta, phi, psi, t0):
        """
        A method for computing the AL displacement of an object in the sky
        at all observation times given its 5 parameter astrometric solutions.
        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]
        pmra : float
            proper motion in the RA* direction of the object [mas/yr]
        pmdec : float
            proper motion in the DEC direction of the object [mas/yr]
        dist : float
            distance to the object [pc]
        r_sep : float
            separation between the two object [AU]
        mass : float
            mass of the luminous star [M_sun]
        m_dark: float
            mass of the dark companion [M_sun]
        ecc : float (0 <= ecc < 1)
            eccentricity of the orbit
        theta, phi, psi : float (0 <= theta < 2pi)
            3 Euler angles
        t0 : float (0 <= t0 < 2pi)
            initial position in the orbit
        Returns
        -------
        Array of along scan displacements in mas
        """

        ra_ar, dec_ar = self._trajectory(ra0, dec0, pmra, pmdec, dist)
        binary_ra, binary_dec = self._binary_correction(r_sep, mass, m_dark, ecc, theta, phi, psi, t0)
        binary_ra = binary_ra*const.au/const.parsec/dist/const.arcsec*1000
        binary_dec = binary_dec*const.au/const.parsec/dist/const.arcsec*1000

        middleIndex = int(np.around((len(self.t_obs) - 1)/2))

        loc = np.array([ra_ar+binary_ra,dec_ar+binary_dec])

        loc_subtracted = np.array([loc[0]-loc[0][middleIndex],loc[1]-loc[1][middleIndex]])

        return self._al_disp(loc_subtracted[0], loc_subtracted[1])

    def binary_2d(self, ra0, dec0, pmra, pmdec, dist, r_sep, mass, m_dark, ecc, theta, phi, psi, t0):
        """
        A method for computing the AL displacement of an object in the sky
        at all observation times given its 5 parameter astrometric solutions.
        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]
        pmra : float
            proper motion in the RA* direction of the object [mas/yr]
        pmdec : float
            proper motion in the DEC direction of the object [mas/yr]
        dist : float
            distance to the object [pc]
        r_sep : float
            separation between the two object [AU]
        mass : float
            mass of the luminous star [M_sun]
        m_dark: float
            mass of the dark companion [M_sun]
        ecc : float (0 <= ecc < 1)
            eccentricity of the orbit
        theta, phi, psi : float (0 <= theta < 2pi)
            3 Euler angles
        t0 : float (0 <= t0 < 2pi)
            initial position in the orbit
        Returns
        -------
        Array of along scan displacements in mas
        """

        ra_ar, dec_ar = self._trajectory(ra0, dec0, pmra, pmdec, dist)
        binary_ra, binary_dec = self._binary_correction(r_sep, mass, m_dark, ecc, theta, phi, psi, t0)
        binary_ra = binary_ra*const.au/const.parsec/dist/const.arcsec*1000
        binary_dec = binary_dec*const.au/const.parsec/dist/const.arcsec*1000

        middleIndex = int(np.around((len(self.t_obs) - 1)/2))

        loc = np.array([ra_ar+binary_ra,dec_ar+binary_dec])

        loc_subtracted = np.array([loc[0]-loc[0][middleIndex],loc[1]-loc[1][middleIndex]])

        return loc_subtracted

    def lensed_2d(self, ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                        ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass):
        """
        A method for computing the displacement of an object in the sky in equatorial coordinates
        at all observation times given its 11 parameter lensed astrometric solutions.

        Parameters
        ----------
        ra_s : float
            reference RA of the source [degrees]
        dec_s : float
            reference DEC of the source [degrees]
        pmra_s : float
            proper motion in the RA* direction of the source [mas/yr]
        pmdec_s : float
            proper motion in the DEC direction of the source [mas/yr]
        dist_s : float
            distance to the source [pc]
        ra_l : float
            reference RA of the lens [degrees]
        dec_l : float
            reference DEC of the lens [degrees]
        pmra_l : float
            proper motion in the RA* direction of the lens [mas/yr]
        pmdec_l : float
            proper motion in the DEC direction of the lens [mas/yr]
        dist_l : float
            distance to the lens [pc]
        mass : float
            mass of the lens

        Returns
        -------
        Double of displacements in equatorial coordinates with unit mas
        """

        ra_ar, dec_ar = self._trajectory_lensed(ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                                     ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass)

        return (ra_ar, dec_ar)


    ##### helper methods #####

    def _earth_cart(self):
        """
        A method that returns the earth's position in heliocentric cartesian coordinates (in AU)
        where the x-axis points toward the vernal equinox, z-axis points toward the celestial north pole.
        x-y plane defines the equatorial plane.

        The Earth's motion is assumed to be a elliptical motion and this script agrees
        with the astropy.coordinates.get_body_barycentric function to ~0.1%.

        Note: this is only accurate to first order in eccentricity due to the complexity
              of solving a transcendental equation M = E - ecc sin(E) the Kepler's equation.

        Returns
        -------
        3D array of the heliocentric cartesian position of the earth at time self.t_obs in AU.
        """

        t_parallax = self.t_obs - 0.0075 # there is an offset of ~3 days with the Julian year here
                                         # and the Julian year used in astropy.time
        m = 2*np.pi*t_parallax
        E = m + self._ecc*np.sin(m)
        theta = 2*np.arctan(np.tan(E/2)*np.sqrt((1+self._ecc)/(1-self._ecc))) - self._theta_eq
        r = 1 - self._ecc*np.cos(E)
        x = -r*np.cos(theta)
        y = -r*np.sin(theta)
        z = np.zeros(self.n_obs)

        return self._obliq_rotation.dot([x,y,z])


    def _parallax(self, ra0, dec0):
        """
        A method to calculate the parallax motion of a souce located at (ra, dec) at 1 kpc.

        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]

        Returns
        -------
        2D array of the parallax motion projected onto (RA, DEC) plane
        """

        ra = ra0*const.degree
        dec = dec0*const.degree
        j = np.array([[np.sin(ra), -np.cos(ra), 0],
                      [np.cos(ra)*np.sin(dec), np.sin(ra)*np.sin(dec), -np.cos(dec)]])

        return j.dot(self.earth_loc)


    def _trajectory(self, ra0, dec0, pmra, pmdec, dist):
        """
        A method to calculate the 2D trjectory of an object.

        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]
        pmra : float
            proper motion in the RA* direction of the object [mas/yr]
        pmdec : float
            proper motion in the DEC direction of the object [mas/yr]
        dist : float
            distance to the object [pc]

        Returns
        -------
        tuple of arrays of the trajectory of the object given the five astrometric
        solutions.
        """

        p = self._parallax(ra0, dec0)/dist*1010.0     # parallax motion
        ra_ar = pmra*(self.t_obs - self.t_ref) + p[0]
        dec_ar = pmdec*(self.t_obs - self.t_ref) + p[1]
        return ra_ar, dec_ar


    def _trajectory_7p(self, ra0, dec0, pmra, pmdec, dist, accra, accdec):
        """
        A method to calculate the 2D trjectory of an object.

        Parameters
        ----------
        ra0 : float
            reference RA of the object [degrees]
        dec0 : float
            reference DEC of the object [degrees]
        pmra : float
            proper motion in the RA* direction of the object [mas/yr]
        pmdec : float
            proper motion in the DEC direction of the object [mas/yr]
        dist : float
            distance to the object [pc]
        accra : float
            proper accleration in the RA* direction of the object [mas/yr^2]
        accdec : float
            proper acceleration in the DEC direction of the object [mas/yr^2]

        Returns
        -------
        tuple of arrays of the trajectory of the object given the five astrometric
        solutions.
        """
        t = self.t_obs - self.t_ref
        p = self._parallax(ra0, dec0)/dist*1010.0     # parallax motion
        ra_ar = .5*accra*t**2 + pmra*t + p[0]
        dec_ar = .5*accdec*t**2 + pmdec*t + p[1]
        return ra_ar, dec_ar

    def _trajectory_lensed(self, ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                                 ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass):
        """
        A method to calculate the 2D trjectory of an object that is deflected by
        a lens.

        Parameters
        ----------
        ra_s : float
            reference RA of the source [degrees]
        dec_s : float
            reference DEC of the source [degrees]
        pmra_s : float
            proper motion in the RA* direction of the source [mas/yr]
        pmdec_s : float
            proper motion in the DEC direction of the source [mas/yr]
        dist_s : float
            distance to the source [pc]
        ra_l : float
            reference RA of the lens [degrees]
        dec_l : float
            reference DEC of the lens [degrees]
        pmra_l : float
            proper motion in the RA* direction of the lens [mas/yr]
        pmdec_l : float
            proper motion in the DEC direction of the lens [mas/yr]
        dist_l : float
            distance to the lens [pc]
        mass : float
            mass of the lens

        Returns
        -------
        tuple of arrays of the trajectory of the object given the 11 parameters
        lensed astrometric solution.
        """

        # Calculate the source trajectory
        s_ra, s_dec = self._trajectory(ra_s, dec_s, pmra_s, pmdec_s, dist_s)
        # Calculate the lens trajectory
        l_ra, l_dec = self._trajectory(ra_l, dec_s, pmra_l, pmdec_l, dist_l)

        eins_r = _eins_r(mass, dist_s, dist_l)

        # impact parameter in the RA* direction [mas]
        b_ra = s_ra - l_ra + (ra_s - ra_l)*np.cos(dec_s*const.degree)*3600*1000
        # impact parameter in the DEC direction [mas]
        b_dec = s_dec - l_dec + (dec_s - dec_l)*3600*1000

        u_ra = b_ra/eins_r
        u_dec = b_dec/eins_r
        u_sq = u_ra**2 + u_dec**2

        # Calculate the angular deflection
        d_ra = eins_r*u_ra/(u_sq + 2)
        d_dec = eins_r*u_dec/(u_sq + 2)

        s_ra_lensed = s_ra + d_ra
        s_dec_lensed = s_dec + d_dec

        return s_ra_lensed, s_dec_lensed


    def _traj_lensed_mag(self, ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                                 ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass):
        """
        A method to calculate the 2D trjectory of an object that is deflected by
        a lens.

        Parameters
        ----------
        ra_s : float
            reference RA of the source [degrees]
        dec_s : float
            reference DEC of the source [degrees]
        pmra_s : float
            proper motion in the RA* direction of the source [mas/yr]
        pmdec_s : float
            proper motion in the DEC direction of the source [mas/yr]
        dist_s : float
            distance to the source [pc]
        ra_l : float
            reference RA of the lens [degrees]
        dec_l : float
            reference DEC of the lens [degrees]
        pmra_l : float
            proper motion in the RA* direction of the lens [mas/yr]
        pmdec_l : float
            proper motion in the DEC direction of the lens [mas/yr]
        dist_l : float
            distance to the lens [pc]
        mass : float
            mass of the lens

        Returns
        -------
        tuple of arrays of the trajectory of the object given the 11 parameters
        lensed astrometric solution.
        """

        # Calculate the source trajectory
        s_ra, s_dec = self._trajectory(ra_s, dec_s, pmra_s, pmdec_s, dist_s)
        # Calculate the lens trajectory
        l_ra, l_dec = self._trajectory(ra_l, dec_s, pmra_l, pmdec_l, dist_l)

        eins_r = _eins_r(mass, dist_s, dist_l)

        # impact parameter in the RA* direction [mas]
        b_ra = s_ra - l_ra + (ra_s - ra_l)*np.cos(dec_s*const.degree)*3600*1000
        # impact parameter in the DEC direction [mas]
        b_dec = s_dec - l_dec + (dec_s - dec_l)*3600*1000

        u_ra = b_ra/eins_r
        u_dec = b_dec/eins_r
        u_sq = u_ra**2 + u_dec**2

        mag = (u_sq + 2)/np.sqrt(u_sq*(u_sq + 4))
        mag = -2.5*np.log10(mag)

        # Calculate the angular deflection
        d_ra = eins_r*u_ra/(u_sq + 2)
        d_dec = eins_r*u_dec/(u_sq + 2)

        s_ra_lensed = s_ra + d_ra
        s_dec_lensed = s_dec + d_dec

        return s_ra_lensed, s_dec_lensed, mag

    def _binary_correction(self, r_sep, mass, m_dark, ecc, theta, phi, psi, t0):
        """
        A method that calculates the binary motion from a dark companion
        Parameters
        ----------
        r_sep : float
            separation between the two object [AU]
        mass : float
            mass of the luminous star [M_sun]
        m_dark: float
            mass of the dark companion [M_sun]
        ecc : float (0 <= ecc < 1)
            eccentricity of the orbit
        theta, phi, psi : float (0 <= theta < 2pi)
            3 Euler angles
        t0 : float (0 <= t0 < 2pi)
            initial position in the orbit
        Returns
        -------
        tuple of arrays of the trajectory correction of the object given the 8
        parameters for binary orbit
        """
        M_tot = mass + m_dark
        n = np.sqrt(const.G*_M_sun*M_tot/(r_sep*const.au)**3)*const.year

        E_ar = np.zeros(self.n_obs)
        for i in range(self.n_obs):
            E_ar[i] = op.newton(_anomaly, 0, args = (ecc, n*self.t_obs[i], t0))
        x = np.cos(E_ar) - ecc

        b = np.sqrt(1 - ecc**2)
        y = b*np.sin(E_ar)
        r_vec = np.array([x,y])
        semi_major = r_sep*m_dark/M_tot/(1 - ecc)
        rot_vec = _rot_2d(theta, phi, psi).dot(r_vec)*semi_major

        ra = rot_vec[0]
        dec = rot_vec[1]
        return ra, dec

    def _trajectory_lensed_extend(self, ra_s, dec_s, pmra_s, pmdec_s, dist_s,
                                  ra_l, dec_l, pmra_l, pmdec_l, dist_l, mass, rs):
        """
        A method to calculate the 2D trjectory of an object that is deflected by
        a lens.
        Parameters
        ----------
        ra_s : float
            reference RA of the source [degrees]
        dec_s : float
            reference DEC of the source [degrees]
        pmra_s : float
            proper motion in the RA* direction of the source [mas/yr]
        pmdec_s : float
            proper motion in the DEC direction of the source [mas/yr]
        dist_s : float
            distance to the source [pc]
        ra_l : float
            reference RA of the lens [degrees]
        dec_l : float
            reference DEC of the lens [degrees]
        pmra_l : float
            proper motion in the RA* direction of the lens [mas/yr]
        pmdec_l : float
            proper motion in the DEC direction of the lens [mas/yr]
        dist_l : float
            distance to the lens [pc]
        mass : float
            mass of the lens
        rs : float
            scale radius of the lens density profile (gaussian density profile) [pc]
        Returns
        -------
        tuple of arrays of the trajectory of the object given the 11 parameters
        lensed astrometric solution.
        """

        # Calculate the source trajectory
        s_ra, s_dec = self._trajectory(ra_s, dec_s, pmra_s, pmdec_s, dist_s)
        # Calculate the lens trajectory
        l_ra, l_dec = self._trajectory(ra_l, dec_s, pmra_l, pmdec_l, dist_l)

        #eins_r = _eins_r(mass, dist_s, dist_l)

        # impact parameter in the RA* direction [mas]
        b_ra = s_ra - l_ra + (ra_s - ra_l)*np.cos(dec_s*const.degree)*3600*1000
        # impact parameter in the DEC direction [mas]
        b_dec = s_dec - l_dec + (dec_s - dec_l)*3600*1000

        b_sq = b_ra**2 + b_dec**2

        r_impact = dist_l*np.sqrt(b_sq)/3600/1000*const.degree
        #print(r_impact)
        m_frac = m_enclose(r_impact/rs)

        #mass = m_column(rs, rho, r_impact)
        #print(m_frac)
        eins_r = _eins_r(mass*m_frac, dist_s, dist_l)

        u_ra = b_ra/eins_r
        u_dec = b_dec/eins_r
        u_sq = u_ra**2 + u_dec**2

        # Calculate the angular deflection
        d_ra = eins_r*u_ra/(u_sq + 2)
        d_dec = eins_r*u_dec/(u_sq + 2)

        s_ra_lensed = s_ra + d_ra
        s_dec_lensed = s_dec + d_dec

        return s_ra_lensed, s_dec_lensed

    def _al_disp(self, ra_disp, dec_disp):
        """
        A method for computing the along displacement of an object in the sky
        at all observation times given its equatorial coordinates.

        Parameters
        ----------
        ra_disp : array/pd.Series
            Object linearized RA* displacemnet from the reference point at t_ref [mas]
        dec_disp : array/pd.Series
            Object DEC displacemnet from the reference point at t_ref [mas]

        Preconditions
        -------------
        len(ra_disp) = len(dec_disp) = len(self.n_obs)

        Returns
        -------
        Array of along scan displacements in mas
        """

        assert len(ra_disp) == self.n_obs, "RA array does not have the same length as n_obs"
        assert len(dec_disp) == self.n_obs, "RA array does not have the same length as n_obs"

        disp = np.sin(self.scan_angles)*ra_disp\
                + np.cos(self.scan_angles)*dec_disp # Along direction displacement [mas]

        return disp


##### helper functions #####

def _eins_r(mass, d_s, d_l):
    """
    A function that calculates the einstein radius

    Parameters
    ----------
    mass : float
        mass of the lens [solar mass]
    d_s : float
        distance to the source [pc]
    d_l : float
        distance to the lens [pc]

    Returns
    -------
    einstein radius in mas
    """

    return 1/const.arcsec*1000*np.sqrt(4*const.G*mass/const.c**2*_M_sun*(d_s -
                                                d_l)/d_l/(d_s*const.parsec))

def rho_exp(x):
    """
    A function that returns the integrand for calculating the column mass for
    gravitational lensing from extended lens that follows a gaussian density profile
    Parameters
    ----------
    x : float/array
        impact parameter to lens [r/r_scale]
    Returns
    -------
    integrand in the following form:
        x/2 exp(-x^2/4) K_0(x^2/4)
    """
    return x*np.exp(-x**2/4)*sp.kn(0,x**2/4)/2


def m_enclose(b):
    """
    A function that calculates the fractional mass enclose in the cylinder with
    impact parameter b
    Parameters
    ----------
    b : array
        impact parameter to lens [r/r_scale]
    Returns
    -------
    fractional mass enclose
    """
    m_frac = np.zeros(len(b))
    for i in range(len(b)):
        m_frac[i] = integrate.quad(rho_exp, 0, b[i])[0]
    return m_frac


def _anomaly(x, ecc, M, t0):
    """
    A function used for newton mehthod to solve for the eccentricity anomaly
    of an eccentric orbit using Kepler's equation
    Parameters
    ----------
    x : float
        eccentricity anomaly
    ecc : float (0 <= ecc < 1)
        eccentricity of the orbit
    M : float
        mean anomaly of the orbit
    t0 : float (0 <= t0 < 2pi)
        initial position of mean anomaly
    """
    return x - ecc*np.sin(x) - M + t0

def _rot_2d(theta, phi, psi):
    """
    A rotational matrix to rotate an orbital plane into arbitrary orientation
    in the celestial sphere
    Parameters
    ----------
    theta, phi, psi : float (0 <= theta < 2pi)
        3 Euler angles
    """
    return np.array([[np.cos(theta)*np.cos(phi), np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)],
                     [np.cos(theta)*np.sin(phi), np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi)],
                     [-np.sin(theta), np.sin(psi)*np.cos(theta)]])

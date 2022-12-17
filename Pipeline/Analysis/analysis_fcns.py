# This script contains all functions for carrying out the statistical search for blips in the GAIA data.

import numpy as np
from pygaia.errors import astrometric as pg_err
import galpy.util.coords as gp_coord
import pandas as pd

import healpy as hp
r = hp.rotator.Rotator(coord = ('C', 'G'))

import scipy.constants as const

from astropy import units as u
from astropy.coordinates import SkyCoord

import os
import sys

# Go to parent directory for fcn imports
os.chdir('..')
cwd = os.getcwd()
sys.path.insert(0, cwd)
import dynamics_fcns
import bh_prior_fcns

# Load file containing time and scan angle information for all observations
obs_info = pd.read_csv('./obs_info.csv', sep=",", header=None, skiprows = [0])
obs_info.columns = ['t_obs','scan_angles']
t_obs = obs_info['t_obs'].to_numpy() # Observation times [Julian years]
scan_angles = obs_info['scan_angles'].to_numpy() #  Scan angles [rad]
t_ref = 2017.5 # Reference time for calculating displacement [Julian years]
n_obs = len(t_obs) # Number of observations

class blip_search():
    """
    A class containing all functions used for mono blip searches.
    """

    def __init__(self,object_type):

        """"
        Class constructor. Sets the type of lens to be searched for (currently only BH).

        Parameters
        ----------
        object_type - The type of lens to be searched for. Currently, these include "black_hole".

        """

        # Initialize dyanmics and priors classes
        self.dynamics = dynamics_fcns.Dynamics() # Initialize the Dyanmics class
        self.priors = bh_prior_fcns.BH_priors() # Initialize the BH_priors class

    def disp_err(self,gmag): #
        """
        Computes the error bars on along (AL) scan displacement data.

        Paramters
        ---------
        scan_angle : float or array
            the GAIA scan angle for the data point or data points [rad]
        gmag : float
            the g-magnitude of the source being analyzed [no unit]

        Returns
        -------
        Float or array of AL scan displacement errors
        """

        gmag_array = np.array([gmag]) # Convert g-magnitude input to array

        error_fcn = pd.read_csv('./coordinate_error.csv', sep=",") # Load the error function csv

        g_mag = error_fcn['G-mag'] # Read the g-magntiude array from the error csv
        err = error_fcn['Error'] # Read the error array from the error csv

        interp = np.interp(gmag_array,g_mag,err)/3 # Interpolate any g-magnitude to an error

        return interp

    def free_2ll(self,al_data,al_err,source_ra0,source_dec0,x):
        """
        Computes the log likelihood for a given data set and a given null (no lensing) model.

        Parameters
        ----------
        al_data : array
            along scan displacement data points for a single event [mas]
        al_err : array
            along scan displacement errors for a single event [mas]
        x : array
            array of float parameters to be passed to the "unlens_AL" function. These are
            ra0 [mas], dec0 [mas], pmra [mas/yr], pmdec [mas/yr], distance [pc]

        Returns
        -------
        Log likelihood of model+data
        """

        # Extract parameters from parameter array
        ra0,dec0,pmra,pmdec,dist = x

        ra0 = source_ra0 # [deg]
        dec0 = source_dec0 # [deg]

        # Compute AL object location based on unlens_AL function
        al_traj =  self.dynamics.unlens_AL(ra0,dec0,pmra,pmdec,dist) # [mas]
        al_offset = np.sin(scan_angles)*x[0]+np.cos(scan_angles)*x[1] # [mas]]

        # Compute difference between data and model AL coordinates
        data_res = al_traj+al_offset-al_data # [mas]

        # Compute the log likelihood of the source motion model
        ll = np.sum((data_res/al_err)**2) # no unit

        return ll

    def free_7p_ll(self,al_data,al_err,source_ra0,source_dec0,x):
        """
        Computes the log likelihood for a given data set and a given null (no lensing) model
        with proper accceleration added.

        Parameters
        ----------
        al_data : array
            along scan displacement data points for a single event [mas]
        al_err : array
            along scan displacement errors for a single event [mas]
        x : array
            array of float parameters to be passed to the "unlens_AL" function. These are
            ra0 [mas], dec0 [mas], pmra [mas/yr], pmdec [mas/yr], distance [pc],
            accra [mas/yr^2], accdec [mas/yr^2]

        Returns
        -------
        Log likelihood of model+data
        """

        # Extract parameters from parameter array
        ra0,dec0,pmra,pmdec,dist,accra,accdec = x

        ra0 = source_ra0 # [deg]
        dec0 = source_dec0 # [deg]

        # Compute AL object location based on unlens_AL function
        al_traj =  self.dynamics.unlens7p_AL(ra0,dec0,pmra,pmdec,dist,accra,accdec) # [mas]
        al_offset = np.sin(scan_angles)*x[0]+np.cos(scan_angles)*x[1] # [mas]

        # Compute difference between data and model AL coordinates
        data_res = al_traj+al_offset-al_data # [mas]

        # Compute the 2 log likelihood of the source motion model
        ll = np.sum((data_res/al_err)**2) # no unit

        return ll

    def _lensed_2ll(self,al_data,al_err,source_ra0,source_dec0,x):
    
        """
        Computes the log likelihood for a given data set and a given lensing model.

        Parameters
        ----------
        al_data : array
             along scan displacement data points for a single event [mas]
        al_err : array
             along scan displacement errors for a single event [mas]
        x : array
            array of float parameters to be passed to the "lensed_AL" function. These are
            ra_s [mas], dec_s [mas], pmra_s [mas/yr], pmdec_s [mas/yr], dist_s [pc],
            ra_l [mas], dec_l [mas], pmra_l [mas/yr], pmdec_l [mas/yr], dist_l [pc],
            mass [solar masses].

        Returns
        -------
        Log likelihood of model+data

        """
        # Extract model parameters from parameter array
        ra_s,dec_s,pmra_s,pmdec_s,dist_s,ra_l,dec_l,pmra_l,pmdec_l,dist_l,mass = x

        d_ra_l = (ra_l-ra_s)/3600/1000/np.cos(source_dec0*np.pi/180) # [deg]
        d_dec_l = (dec_l-dec_s)/3600/1000 # [deg]

        # Compute the AL location of the source according to the lensed model
        al_traj = self.dynamics.lensed_AL(source_ra0,source_dec0,pmra_s,pmdec_s,dist_s,
            source_ra0+d_ra_l,source_dec0+d_dec_l,pmra_l,pmdec_l,dist_l,mass) # [mas]

        # Compute the AL offset
        al_offset = np.sin(scan_angles)*x[0]+np.cos(scan_angles)*x[1] # [mas]

        # Compute the difference in AL coordinates between the data and the unlensed model
        data_res = al_traj+al_offset-al_data # [mas]

        # Compute th two log likelihood of the lensing model
        ll = np.sum(((data_res)/al_err)**2) # no unit

        return ll
        
    def llr(self,al_data,al_err,ll_null,source_ra0,source_dec0,x):
        """
        Computes the log likelihood ratio of two models (typically signal and null)

        Parameters
        ----------
        al_data : array
            along scan displacement data points for a single event [mas]
        al_err : array
            along scan displacement errors for a single event [mas]
        ll_null : float
            the log likelihood of the null model
        x : array
            array of float parameters to be passed to the "_lensed_2ll" function. These are
            ra_s [mas], dec_s [mas], pmra_s [mas/yr], pmdec_s [mas/yr], dist_s [pc],
            ra_l [mas], dec_l [mas], pmra_l [mas/yr], pmdec_l [mas/yr], dist_l [pc],
            mass [solar masses].

        Returns
        -------
        Two log likelihood ratio of the two models.
        """

        # Compute the signal log likelihood
        ll_signal = self._lensed_2ll(al_data,al_err,source_ra0,source_dec0,x) # no unit

        # Compute the log likelihood ratio between the null and the signal
        twollr = ll_signal-ll_null # no unit

        return twollr

    def ts_priors(self,al_data,al_err,source_ra0,source_dec0,x):
        """
        Computes the log likelihood ratio of two models (typically signal and null)

        Parameters
        ----------
        al_data : array
            along scan displacement data points for a single event [mas]
        al_err : array
            along scan displacement errors for a single event [mas]
        ll_null : float
            the log likelihood of the null model
        x : array
            array of float parameters to be passed to the "_lensed_2ll" function. These are
            ra_s [mas], dec_s [mas], pmra_s [mas/yr], pmdec_s [mas/yr], dist_s [pc],
            ra_l [mas], dec_l [mas], pmra_l [mas/yr], pmdec_l [mas/yr], dist_l [pc],
            mass [solar masses].

        Returns
        -------
        Two log likelihood ratio of the two models.
        """

        b, l = r((90 - source_dec0)*const.degree, source_ra0*const.degree)/const.degree
        b = 90 - b

        log_joint_prob = np.log(self.priors.joint_pdf_BH(x[7],x[8],x[9]/1000.,l,b)) # Compute the log of the joint
        log_mass_prob = np.log(self.priors.mass_BH(x[10])) #

        # Compute the signal log likelihood
        ts_signal = self._lensed_2ll(al_data,al_err,source_ra0,source_dec0,x)-2*log_joint_prob-2*log_mass_prob # no unit

        return ts_signal

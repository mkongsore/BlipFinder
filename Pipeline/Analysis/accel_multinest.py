# GAIA Blip Analysis Script
# Calvin Chen, Marius Kongsore, Ken Van Tilburg
# 2022

# This file as part of a larger GitHub package
# https://github.com/mkongsore/gaiablip

############################
# Load Functions and Files #
############################

import os
import sys
os.chdir('/home/mk7976/git/gaiablip') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

results_path = '/scratch/mk7976/fit_results/x1_new/accel_multinest_results'
catalog_path = '/scratch/mk7976/epoch_astrometry/lens_new' # Specify the folder to load the catalog from
postsamples_path = '/scratch/mk7976/fit_results/x1_new/accel_postsamples'

import dynamics_fcns as df
import bh_prior_fcns

priors = bh_prior_fcns.BH_priors()
dynamics = df.Dynamics()

# Change system path
os.chdir('/home/mk7976/git/gaiablip/analysis/') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

import numpy as np
import matplotlib.pyplot as plt

import analysis_fcns as af
import pandas as pd

import scipy

# import PyMultiNest Solver class
from pymultinest.solve import Solver

import constraint_fcns # Import the constraint functions script
free_constraints = constraint_fcns.cons_free # Tuple of functions contraining blip model fitting

obs_info = pd.read_csv('./obs_info.csv', sep=",", header=None,skiprows=[0]) # Read observation info csv
obs_info.columns = ['t_obs','scan_angles'] # Specify what each column in obs_info file are
# Extrapolate info from obs_info file
t_obs = obs_info['t_obs'].to_numpy() # Observation times [Julian years]
scan_angles = obs_info['scan_angles'].to_numpy() #  Scan angles [rad]
t_ref = 2017.5 # Reference time for calculating displacement [Julian years]
n_obs = len(t_obs) # Number of observations
dof = n_obs-5

blip_search = af.blip_search('bh')
catalog_list = os.listdir('/scratch/mk7976/epoch_astrometry/lens_new')

os.chdir('/home/mk7976/git/gaiablip/') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

# Specify the minimum unlensed 2LL that a source must have to be saved
job_idx = int(sys.argv[1])

catalog_name = os.listdir(catalog_path)[job_idx] # Pick the file name corresponding to the job to be ananalyzed
file_number = catalog_name[18:31] # Pick the data file number from the data file name
catalog_info_name = 'gaia_info_'+file_number+'.csv' # Anitialize the name of the info file correpsponding to the data file

data = pd.read_pickle('/scratch/mk7976/epoch_astrometry/lens_new/'+catalog_name) 
catalog_id_list = list(data['source_id'])

misc_info_folder = '/scratch/ic2127/gaia_edr3_info/' # Specify the location of the folder containing the file with parallax and g magnitude data
misc_info_data = pd.read_csv(misc_info_folder+catalog_info_name) # Load in file containing parallax and g magnitude information

results = pd.read_csv('/scratch/mk7976/fit_results/x1_new/accel_fit_results/accel_'+file_number+'.csv')

# Load random seed list 
seed_info_folder = './analysis/seed_lists/'
seed_info = pd.read_csv('/scratch/mk7976/seed_lists/'+file_number+'_seeds.csv')

first_sig_event = True

def inverse_gaussian_cdf(x,mu,sigma): # Define the inverse gaussian cdf function to be used for inverse transform sampling
    val = np.sqrt(2)*sigma*scipy.special.erfinv(2*x-1)+mu
    return val

for n in range(np.size(results['ts'])):   

    s_id = results.iat[n,0]
    results_row = results.iloc[n]
    s_idx = catalog_id_list.index(int(s_id))
    s_row = data.iloc[s_idx] # Pick particular RA row corresponding to source of interest
    s_info_row = misc_info_data.iloc[s_idx] # Pick the particular data file row corresponding to the source of interest


    np.random.seed(int(seed_info.iat[s_idx,2])) # Set a random seed to ensure data gets scrambled in the same way every time

    s_accel_ra0 = float(results_row[1])
    s_accel_dec0 = float(results_row[2])
    s_accel_pmra = float(results_row[3])
    s_accel_pmdec = float(results_row[4])
    s_accel_dist = float(results_row[5])
    s_accel_acra = float(results_row[6])
    s_accel_acdec = float(results_row[7])

    s_ra0 = float(s_row[1]) # RA of source at first observation epoch [deg]
    s_dec0 = float(s_row[2]) # DEC of source at first observation epoch [deg]
    s_ddisp_noerr = np.array(s_row[3:]) # Change in RA [mas]
    s_dist = float(s_info_row[3]) # Estimated distance to source [pc]
    s_gmag = float(s_info_row[6]) # G magntiude of source

    s_ddisp_err = blip_search.disp_err(s_gmag)
    s_ddisp = s_ddisp_noerr
    s_ddisp = np.random.normal(loc = s_ddisp_noerr,scale = s_ddisp_err) # Scramble data according to a normal distribution with 1 sigma = source_ddec_err [mas]

    x = np.array([s_accel_ra0,s_accel_dec0,s_accel_pmra,s_accel_pmdec,s_accel_dist,s_accel_acra,s_accel_acdec])

    print(x)

    if True: # Do lensed fit if greater than five sigma chi_sq


##################################################

        print('commencing new fit')

        # Create Solver class
        class AccelModelPyMultiNest(Solver):
            """
            The lensed model, with a Gaussian likelihood.

            Args:
                data (:class:`numpy.ndarray`): an array containing the observed data
                modelfunc (function): a function defining the model
                sigma (float): the standard deviation of the noise in the data
                **kwargs: keyword arguments for the run method
            """
            
            def __init__(self, data, sigma, **kwargs):
                # set the data
                
                self._data = data         # oberserved data
                self._sigma = sigma       # standard deviation(s) of the data
                self._logsigma = np.log(self._sigma) # log sigma here to save computations in the likelihood
                self._ndata = len(data)   # number of data points

                Solver.__init__(self, **kwargs)

            def Prior(self, cube):
                """
                The prior transform going from the unit hypercube to the true parameters. This function
                has to be called "Prior".

                Args:
                    cube (:class:`numpy.ndarray`): an array of values drawn from the unit hypercube

                Returns:
                    :class:`numpy.ndarray`: an array of the transformed parameters
                """

                # extract values
                self.s_raprime = cube[0]
                self.s_decprime = cube[1]
                self.s_pmraprime = cube[2]
                self.s_pmdecprime = cube[3]
                self.s_distprime = cube[4]
                self.s_acraprime = cube[5]
                self.s_acdecprime = cube[6]
            
                params = cube.copy()
                
                params[0] = inverse_gaussian_cdf(self.s_raprime,x[0],20.)
                params[1] = inverse_gaussian_cdf(self.s_decprime,x[1],20.)
                params[2] = inverse_gaussian_cdf(self.s_pmraprime,x[2],30.)
                params[3] = inverse_gaussian_cdf(self.s_pmdecprime,x[3],30.)
                params[4] = np.absolute(inverse_gaussian_cdf(self.s_distprime,np.min([np.absolute(x[4]),5000]),5000.))
                params[5] = inverse_gaussian_cdf(self.s_acraprime,x[5],200)
                params[6] = inverse_gaussian_cdf(self.s_acdecprime,x[6],200)

                return params

            def LogLikelihood(self, cube):
                """
                The log likelihood function. This function has to be called "LogLikelihood".

                Args:
                    cube (:class:`numpy.ndarray`): an array of parameter values.

                Returns:
                    float: the log likelihood value.
                """

                # extract values
                s_ra = cube[0]
                s_dec = cube[1]
                s_pmra = cube[2]
                s_pmdec = cube[3]
                s_dist = cube[4]
                s_acra = cube[5]
                s_acdec = cube[6]
                
                parms = np.array([s_ra,s_dec,s_pmra,s_pmdec,s_dist,s_acra,s_acdec])

                # calculate the model
                
                ll = -blip_search.free_7p_ll(self._data,self._sigma,s_ra0,s_dec0,parms)

                if np.isnan(ll)==True or np.isinf(ll)==True or s_dist<=0: # Handle case where ll violates prior
                    ll = -1.0e100                
 
                return ll

        nlive = 700 # number of live points
        ndim = 7     # number of parameters
        tol = 0.1    # stopping criterion


        # run the algorithm
        solution = AccelModelPyMultiNest(data=s_ddisp, sigma=s_ddisp_err, n_dims=ndim,
                                                n_live_points=nlive, evidence_tolerance=tol,
                                                resume = False,verbose=True)

        logZpymnest = solution.logZ        # value of log Z
        logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

        s_rachain_pymnest = solution.samples[:,0] # extract chain of ra values
        s_decchain_pymnest = solution.samples[:,1] # extract chain if dec values
        s_pmrachain_pymnest = solution.samples[:,2] # extract chain of ra values
        s_pmdecchain_pymnest = solution.samples[:,3] # extract chain if dec values
        s_distchain_pymnest = solution.samples[:,4] # extract chain of ra values
        s_raacchain_pymnest = solution.samples[:,5] # extract chain of ra values
        s_decacchain_pymnest = solution.samples[:,6] # extract chain of ra values

        postsamples = np.vstack((s_rachain_pymnest, s_decchain_pymnest,s_pmrachain_pymnest,s_pmdecchain_pymnest,s_distchain_pymnest,
                                s_raacchain_pymnest,s_decacchain_pymnest)).T

        y_bf = solution.samples[-1] # Save bf parameters

        blip_fit = scipy.optimize.minimize(lambda x: blip_search.free_7p_ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
                                     x0=y_bf, # Specify the initial guess
                                     method = 'SLSQP', # Select Sequential Least SQuares Programming based minimizer
                                     tol= 1e-7, # Set the tolarance level for what is considered a minimum
                                     jac = '3-point', # Set the Jacobian to be of the 3-point type
                                     options = {'maxiter':1000000}, # Set the miximum number of minimizer iterations
                                     )

        bf_ts = blip_search.free_7p_ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,blip_fit.x) # Save the test statistic obtained from the fit
        bf_x = blip_fit.x # Save the best fit parameters obtained from the fit

        print('bf ts',bf_ts)

        ################
        # Save Results #
        ################

        # Save the best fit parameters and source id to a dictionary
        data_out = {'id':[s_id],'x0':[bf_x[0]],'x1':[bf_x[1]],'x2':[bf_x[2]],
                    'x3':[bf_x[3]],'x4':[bf_x[4]],'x5':[bf_x[5]],'x6':[bf_x[6]],
                    'ts':[bf_ts]}

        # Convert the output dictionary to a pandas dataframe
        dataf = pd.DataFrame(data_out)

        # Save the output to a csv file corresponding to the datafile the source is in to scratch
        if first_sig_event==True: # Save w/ header if first >5sigma source
            dataf.to_csv(results_path+'/accel_'+str(file_number)+'.csv', mode='a', index=False, header=(('s_id','s_free_delta_ra0 [mas]','s_free_delta_dec0 [mas]','s_free_pm_ra [mas/yr]','s_free_pm_dec[mas/yr]','s_free_dist [pc]','s_accel_ra [mas/yr/yr]','s_accel_dec [mas/yr/yr]','ts')))
            first_sig_event = False

        else: # Else append without header
            # Save the output to a csv file corresponding to the datafile the source is in to scratch
            dataf.to_csv(results_path+'/accel_'+str(file_number)+'.csv', mode='a', index=False, header=False)

        # Save the postsamples in every case
        np.savez(postsamples_path+'/post_'+str(file_number)+'_'+str(s_id)+'.npz',postsamples=postsamples)

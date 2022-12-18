# GAIA Blip Model Script
# I-Kai Chen, Marius Kongsore, Ken Van Tilburg
# December 2022

# This file as part of a larger GitHub package
# https://github.com/mkongsore/BlipFinder

############################
# Load Functions and Files #
############################

# Import system packages
import sys
import os

# Import public packages
import scipy 
import pandas as pd
import numpy as np
from pymultinest.solve import Solver

# Import BlipFinder analysis functions
import analysis_fcns # Import the analysis functions script
import constraint_fcns # Import the constraint functions script

# Initialize functions for statistical analysis ('black_holes' is a placeholder)
analysis = analysis_fcns.blip_search('black_holes')

# Initialize constraint functions
blip_constraints = constraint_fcns.cons_blip # Tuple of functions contraining blip model fitting

# Import fcns scripts
import dynamics_fcns # Import the dynamics functions script
import bh_prior_fcns # Import the black hole prior functions script

# Change system path
sys.path.append('../')

# Import BlipFinder dynamics and prior functions
dynamics = dynamics_fcns.Dynamics() # Functions for stellar and lens motion
priors = bh_prior_fcns.BH_priors() # Functions for black hole priors

print('Functions Initialized')

# Retrieve an integer between 0 and the number of GAIA files-1 as specified by the batch script
job_idx = 0 #int(sys.argv[1])

# Load file containing time and scan angle information for all observations
obs_info = pd.read_csv('./obs_info.csv', sep=",") # Read observation info csv

# Obtain general Gaia observation info from obs_info file
t_obs = obs_info['Observation Times [Julian Year]'].to_numpy() # Observation times [Julian years]
scan_angles = obs_info['Scan Angles [rad]'].to_numpy() #  Scan angles [rad]
t_ref = 2017.5 # Reference time for calculating displacement [Julian years]

file_name = os.listdir('./Data')[job_idx] # Pick the file name corresponding to the job to be analyzed
file_id = str(file_name[11:24]) # Pick the data file number from the data file name
info_file_name = 'gaia_info_'+file_id+'.csv' # Initialize the name of the info file correpsponding to the data file

# Load data file containing the displacement-time coordinates
data = './Data/gaia_epoch_'+file_id+'.pkl' # Specify the location of the obs files
data = pd.read_pickle(data) # Load observation data from pickle file
source_info = pd.read_csv('./SourceInfo/'+info_file_name) # Load in file containing parallax and g magnitude information
catalog_id_list = list(data['source_id']) # List of source IDs in the catalog

# Load random seed list
seed_info = pd.read_csv('./SourceInfo/'+file_id+'_seeds.csv')

# Load the file with the results from the initial fit
free_multinest_results = pd.read_csv('./Results/FreeMultinest/free_'+file_id+'.csv')

first_sig_event = True # Set first significant even condition to True

file_name = os.listdir('./Data')[job_idx] # Pick the file name corresponding to the job to be analyzed
file_id = str(file_name[11:24]) # Pick the data file number from the data file name
info_file_name = 'gaia_info_'+file_id+'.csv' # Initialize the name of the info file correpsponding to the data file

# Load data file containing the displacement-time coordinates
data = './Data/gaia_epoch_'+file_id+'.pkl' # Specify the location of the obs files
data = pd.read_pickle(data) # Load observation data from pickle file
source_info = pd.read_csv('./SourceInfo/'+info_file_name) # Load in file containing parallax and g magnitude information
catalog_id_list = list(data['source_id']) # List of source IDs in the catalog

# Load random seed list
seed_info = pd.read_csv('./SourceInfo/'+file_id+'_seeds.csv')

# Load the file with the results from the initial fit
blip_scipy_results = pd.read_csv('./Results/BlipScipy/blip_'+file_id+'.csv')
free_multinest_results = pd.read_csv('./Results/FreeMultinest/free_'+file_id+'.csv')

first_sig_event = True # Set first significant even condition to True

for n in range(np.size(blip_scipy_results['s_id'])):

    s_id = blip_scipy_results.iat[n,0]
    results_row = blip_scipy_results.iloc[n]

    s_idx = catalog_id_list.index(int(s_id))
    s_row = data.iloc[s_idx] # Pick particular RA row corresponding to source of interest
    s_info_row = source_info.iloc[s_idx] # Pick the particular data file row corresponding to the source of interest

    free_results_row = free_multinest_results.iloc[n]

    s_ra0 = float(s_row[1]) # RA of source at first observation epoch [deg]
    s_dec0 = float(s_row[2]) # DEC of source at first observation epoch [deg]
    s_ddisp_noerr = np.array(s_row[3:]) # Change in RA [mas]
    s_dist = float(s_info_row[3]) # Estimated distance to source [pc]
    s_gmag = float(s_info_row[6]) # G magntiude of source

    if np.isnan(s_gmag)==True:
        s_gmag = 19.571021 # Set G magntiude to the median of the catalog if it is not available

    np.random.seed(int(seed_info.iat[s_idx,2])) # Set a random seed to ensure data gets scrambled in the same way every time

    s_ddisp_err = analysis.disp_err(s_gmag)
    s_ddisp = s_ddisp_noerr
    s_ddisp = np.random.normal(loc = s_ddisp_noerr,scale = s_ddisp_err) # Scramble data according to a normal distribution with 1 sigma = source_ddec_err [mas]
 
    s_blip_ra0 = float(results_row[1])
    s_blip_dec0 = float(results_row[2])
    s_blip_pmra = float(results_row[3])
    s_blip_pmdec = float(results_row[4])
    s_blip_dist = float(results_row[5])

    l_blip_ra0 = float(results_row[6])
    l_blip_dec0 = float(results_row[7])
    l_blip_pmra = float(results_row[8])
    l_blip_pmdec = float(results_row[9])
    l_blip_dist = float(results_row[10])
    l_blip_mass = float(results_row[11])

    x = np.array([s_blip_ra0,s_blip_dec0,s_blip_pmra,s_blip_pmdec,s_blip_dist,l_blip_ra0,l_blip_dec0,l_blip_pmra,l_blip_pmdec,l_blip_dist,l_blip_mass])

    free_ll = float(free_results_row[6])

    if True: # Do lensed fit for all sources in BlipScipy results folder

        print('commencing new fit')

        # Create Solver class
        class BlipModelPyMultiNest(Solver):
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
                self.l_raprime = cube[5]
                self.l_decprime = cube[6]
                self.l_pmraprime = cube[7]
                self.l_pmdecprime = cube[8]
                self.l_distprime = cube[9]
                self.l_massprime = cube[10]
            
                params = cube.copy()
                
                params[0] = analysis.inverse_gaussian_cdf(self.s_raprime,x[0],20.)
                params[1] = analysis.inverse_gaussian_cdf(self.s_decprime,x[1],20.)
                params[2] = analysis.inverse_gaussian_cdf(self.s_pmraprime,x[2],30.)
                params[3] = analysis.inverse_gaussian_cdf(self.s_pmdecprime,x[3],30.)
                params[4] = np.absolute(analysis.inverse_gaussian_cdf(self.s_distprime,np.min([np.absolute(x[4]),5000]),5000.))
                params[5] = analysis.inverse_gaussian_cdf(self.l_raprime,x[5],300.)
                params[6] = analysis.inverse_gaussian_cdf(self.l_decprime,x[6],300.)
                params[7] = analysis.inverse_gaussian_cdf(self.l_pmraprime,x[7],100.)
                params[8] = analysis.inverse_gaussian_cdf(self.l_pmdecprime,x[8],100.)
                params[9] = np.absolute(analysis.inverse_gaussian_cdf(self.s_distprime,np.min([np.absolute(x[9]),5500]),5000.))
                params[10] = np.absolute(analysis.inverse_gaussian_cdf(self.l_massprime,np.min([np.absolute(x[10]),20]),50.))
                
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
                l_ra = cube[5]
                l_dec = cube[6]
                l_pmra = cube[7]
                l_pmdec = cube[8]
                l_dist = cube[9]
                l_mass = cube[10]
                
                parms = np.array([s_ra,s_dec,s_pmra,s_pmdec,s_dist,l_ra,l_dec,l_pmra,l_pmdec,l_dist,l_mass])

                # calculate the model
                
                ll = -analysis.ts_priors(self._data,self._sigma,s_ra0,s_dec0,parms)/2.

                if np.isnan(ll)==True or np.isinf(ll)==True or s_dist<=l_dist or s_dist<=0 or l_dist<=0 or l_mass<=0: # Handle case where ll violates prior
                    ll = -1.0e100                
 
                return ll

        nlive = 700 # number of live points
        ndim = 11     # number of parameters
        tol = 0.1    # stopping criterion

        print('s_gmag',s_gmag)

        # run the algorithm
        solution = BlipModelPyMultiNest(data=s_ddisp, sigma=s_ddisp_err, n_dims=ndim,
                                                n_live_points=nlive, evidence_tolerance=tol,
                                                resume = False,verbose=True)

        logZpymnest = solution.logZ        # value of log Z
        logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

        s_rachain_pymnest = solution.samples[:,0] # extract chain of ra values
        s_decchain_pymnest = solution.samples[:,1] # extract chain if dec values
        s_pmrachain_pymnest = solution.samples[:,2] # extract chain of ra values
        s_pmdecchain_pymnest = solution.samples[:,3] # extract chain if dec values
        s_distchain_pymnest = solution.samples[:,4] # extract chain of ra values
        l_rachain_pymnest = solution.samples[:,5] # extract chain of ra values
        l_decchain_pymnest = solution.samples[:,6] # extract chain if dec values
        l_pmrachain_pymnest = solution.samples[:,7] # extract chain of ra values
        l_pmdecchain_pymnest = solution.samples[:,8] # extract chain if dec values
        l_distchain_pymnest = solution.samples[:,9] # extract chain of ra values
        l_masschain_pymnest = solution.samples[:,10] # extract chain of ra values

        postsamples = np.vstack((s_rachain_pymnest, s_decchain_pymnest,s_pmrachain_pymnest,s_pmdecchain_pymnest,s_distchain_pymnest,
                                l_rachain_pymnest,l_decchain_pymnest,l_pmrachain_pymnest,l_pmdecchain_pymnest,l_distchain_pymnest,l_masschain_pymnest)).T
                
        # MULTINEST END

        y_bf = solution.samples[-1] # Save bf parameters

        blip_fit = scipy.optimize.minimize(lambda x: analysis.ts_priors(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
                                     x0=y_bf, # Specify the initial guess
                                     method = 'SLSQP', # Select Sequential Least SQuares Programming based minimizer
                                     tol= 1e-7, # Set the tolarance level for what is considered a minimum
                                     jac = '3-point', # Set the Jacobian to be of the 3-point type
                                     constraints=blip_constraints, # Load parameter constraints specified by the constraints_fcns script
                                     options = {'maxiter':1000000}, # Set the miximum number of minimizer iterations
                                     )

        bf_ts = analysis.llr(s_ddisp,s_ddisp_err,free_ll,s_ra0,s_dec0,blip_fit.x) # Save the test statistic obtained from the fit
        bf_x = blip_fit.x # Save the best fit parameters obtained from the fit

        ################
        # Save Results #
        ################

        # Save the best fit parameters and source id to a dictionary
        data_out = {'id':[s_id],'x0':[bf_x[0]],'x1':[bf_x[1]],'x2':[bf_x[2]],'x3':[bf_x[3]],'x4':[bf_x[4]],'x5':[bf_x[5]],'x6':[bf_x[6]],'x7':[bf_x[7]],'x8':[bf_x[8]],'x9':[bf_x[9]],'x10':[bf_x[10]],'-ts':[bf_ts]}

        # Convert the output dictionary to a pandas dataframe
        dataf = pd.DataFrame(data_out)

        if first_sig_event==True: # Save w/ header if first >5sigma source
            dataf.to_csv('./Results/BlipMultinest/blip_'+file_id+'.csv', mode='a', index=False, header=(('s_id','s_delta_ra0 [mas]','s_delta_dec0 [mas]','s_pm_ra [mas/yr]','s_pm_dec[mas/yr]','s_dist [pc]','l_delta_ra0 [mas]','l_delta_dec0 [mas]','l_pm_ra [mas/yr]','l_pm_dec[mas/yr]','l_dist [pc]','l_mass [sm]','-ts')))

            first_sig_event = False

        else: # Else append without header
            # Save the output to a csv file corresponding to the datafile the source is in to scratch
            dataf.to_csv('./Results/BlipMultinest/blip_'+file_id+'.csv', mode='a', index=False, header=False)

        # Save the postsamples in every case
        np.savez('./Results/BlipPostsamples/post_'+str(file_id)+'_'+str(s_id)+'.npz',postsamples=postsamples)
    
    elif np.isnan(free_ll)==True:
        print('Fatal Error')
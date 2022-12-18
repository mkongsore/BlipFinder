
# GAIA Free Model Analysis Script
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
import scipy as sp
import pandas as pd
import numpy as np
from pymultinest.solve import Solver

# Import BlipFinder analysis functions
import analysis_fcns # Import the analysis functions script
import constraint_fcns # Import the constraint functions script

# Initialize functions for statistical analysis ('black_holes' is a placeholder)
analysis = analysis_fcns.blip_search('black_holes')

# Initialize constraint functions
free_constraints = constraint_fcns.cons_free # Tuple of functions contraining free model fitting

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

# Load random seed list
seed_info = pd.read_csv('./SourceInfo/'+file_id+'_seeds.csv')

# Load the file with the results from the initial fit
scipy_results = pd.read_csv('./Results/FreeScipy/free_'+file_id+'.csv')

for m in range(np.size(scipy_results['source_id'])):

    s_id = data.iat[m,0] # Source ID
    results_row = scipy_results.iloc[m] # Skip header
    chisq = float(results_row[6]) # chi_square 

    y0 = float(results_row[1])
    y1 = float(results_row[2])
    y2 = float(results_row[3])
    y3 = float(results_row[4])
    y4 = float(results_row[5])

    y=np.array([y0,y1,y2,y3,y4])

    if np.isnan(chisq)==True or np.isnan(y0)==True or np.isnan(y1)==True or np.isnan(y2)==True or np.isnan(y3)==True or np.isnan(y4)==True:

        s_idx = catalog_id_list.index(int(s_id))
        
        s_row = data.iloc[s_idx] # Pick particular RA row corresponding to source of interest
        s_info_row = misc_info_data.iloc[s_idx] # Pick the particular data file row corresponding to the source of interest

        s_ra0 = float(s_row[1]) # RA of source at first observation epoch [deg]
        s_dec0 = float(s_row[2]) # DEC of source at first observation epoch [deg]
        s_ddisp_noerr = np.array(s_row[3:]) # Change in RA [mas]
        s_dist = float(s_info_row[3]) # Estimated distance to source [pc]
        s_gmag = float(misc_info_data.iat[s_idx,6]) # G magntiude of source

        if np.isnan(s_gmag)==True:
            s_gmag = 19.571021 # Set G magntiude to the median of the catalog if it is not available

        np.random.seed(int(seed_info.iat[m,2])) # Set a random seed to ensure data gets scrambled in the same way every time

        s_ddisp_err = analysis.disp_err(s_gmag)
        s_ddisp = np.random.normal(loc = s_ddisp_noerr,scale = s_ddisp_err) # Scramble data according to a normal distribution with 1 sigma = source_ddec_err [mas]

        # create Solver class
        class UnlensedModelPyMultiNest(Solver):
            """
            The lensed model, with a Gaussian likelihood.

            Args:
                data (:class:`numpy.ndarray`): an array containing the observed data
                abscissa (:class:`numpy.ndarray`): an array containing the points at which the data were taken
                modelfunc (function): a function defining the model
                sigma (float): the standard deviation of the noise in the data
                **kwargs: keyword arguments for the run method
            """

            # define the prior parameters
            
            # This is only for the absolute worst case scenario, where there is no best fit for the free model

            ra_min = -100000.
            ra_max = 100000.
            
            dec_min = -100000. 
            dec_max = 100000.
            
            pmra_min = -100000. # Bernard's star (fastest star in the sky) has a proper motion of ~10000 mas/yr
            pmra_max = 100000.
            
            pmdec_min = -100000.
            pmdec_max = 100000.
            
            dist_min = 1. # Proxima Centuri is 1.2948 pc away
            dist_max = 50000.

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
                self.raprime = cube[0]
                self.decprime = cube[1]
                self.pmraprime = cube[2]
                self.pmdecprime = cube[3]
                self.distprime = cube[4]
                
                params = cube.copy()
                
                # Put very wide Guassians around each best fit free parameter as our inverse transform sampling prior
                params[0] = self.raprime*(self.ra_max-self.ra_min)+self.ra_min
                params[1] = self.decprime*(self.dec_max-self.dec_min)+self.dec_min
                params[2] = self.pmraprime*(self.pmra_max-self.pmra_min)+self.pmra_min
                params[3] = self.pmdecprime*(self.pmdec_max-self.pmdec_min)+self.pmdec_min
                params[4] = self.distprime*(self.dist_max-self.dist_min)+self.dist_min

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
                ra = cube[0]
                dec = cube[1]
                pmra = cube[2]
                pmdec = cube[3]
                dist = cube[4]
                
                parms = np.array([ra,dec,pmra,pmdec,dist])

                # calculate the model
                ll = -analysis.free_2ll(self._data,self._sigma,s_ra0,s_dec0,parms)

                if np.isnan(ll)==True or np.isinf(ll)==True or dist<0:
                    ll = -1.0e100
                
                return ll

        nlive = 700 # number of live points
        ndim = 5     # number of parameters
        tol = 0.1    # stopping criterion

        # run the algorithm
        solution = UnlensedModelPyMultiNest(data=s_ddisp, sigma=s_ddisp_err, n_dims=ndim,
                                                n_live_points=nlive, evidence_tolerance=tol,resume=False);

        logZpymnest = solution.logZ        # value of log Z
        logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

        rachain_pymnest = solution.samples[:,0] # extract chain of ra values
        decchain_pymnest = solution.samples[:,1] # extract chain if dec values
        pmrachain_pymnest = solution.samples[:,2] # extract chain of ra values
        pmdecchain_pymnest = solution.samples[:,3] # extract chain if dec values
        distchain_pymnest = solution.samples[:,4] # extract chain of ra values

        postsamples = np.vstack((rachain_pymnest, decchain_pymnest,pmrachain_pymnest,pmdecchain_pymnest,distchain_pymnest)).T

        # MULTINEST END

        y_bf = solution.samples[-1] # Save bf parameters

        # Go the rest of the way home!
        free_fit = scipy.optimize.minimize(lambda x: analysis.free_2ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
                                     x0=y_bf, # Specify the initial guess
                                     method = 'SLSQP', # Select Sequential Least SQuares Programming based minimizer
                                     tol= 1e-7, # Set the tolarance level for what is considered a minimum
                                     jac = '3-point', # Set the Jacobian to be of the 3-point type
                                     constraints=free_constraints, # Load parameter constraints specified by the constraints_fcns script
                                     options = {'maxiter':1000000}, # Set the miximum number of minimizer iterations
                                     )

        y_bf = free_fit.x # Extract best fit parameters
        free_2ll = free_fit.fun # Save the log likelihood of the free fit to the data

        # Save the best fit parameters and source id to a dictionary
        data_out = {'id':[s_id],'y0':[y_bf[0]],'y1':[y_bf[1]],'y2':[y_bf[2]],'y3':[y_bf[3]],'y4':[y_bf[4]],'-2ll':[free_2ll]}

        # Convert the output dictionary to a pandas dataframe
        dataf = pd.DataFrame(data_out)

        if m==0: # Save w/ header if first row
            dataf.to_csv('./Results/FreeMultinest/free_'+file_id+'.csv', mode='a', index=False, header=(('s_id','delta_ra0 [mas]','delta_dec0 [mas]','pm_ra [mas/yr]','pm_dec[mas/yr]','dist [pc]','ts')))

        else: # Else save without header
            dataf.to_csv('./Results/FreeMultinest/free_'+file_id+'.csv', mode='a', index=False, header=False)

        np.savez('./Results/FreePostsamples/post_'+file_id+'.npz',postsamples=postsamples) # Save the postsamples

    elif chisq>152.: # Refit if five sigma chi_square

        s_idx = catalog_id_list.index(int(s_id))
        
        s_row = data.iloc[s_idx] # Pick particular RA row corresponding to source of interest
        s_info_row = misc_info_data.iloc[s_idx] # Pick the particular data file row corresponding to the source of interest

        s_ra0 = float(s_row[1]) # RA of source at first observation epoch [deg]
        s_dec0 = float(s_row[2]) # DEC of source at first observation epoch [deg]
        s_ddisp_noerr = np.array(s_row[3:]) # Change in RA [mas]
        s_dist = float(s_info_row[3]) # Estimated distance to source [pc]
        s_gmag = float(misc_info_data.iat[s_idx,6]) # G magntiude of source

        if np.isnan(s_gmag)==True:
            s_gmag = 19.571021 # Set G magntiude to the median of the catalog if it is not available

        np.random.seed(int(seed_info.iat[m,2])) # Set a random seed to ensure data gets scrambled in the same way every time

        s_ddisp_err = analysis.disp_err(s_gmag)
        s_ddisp = np.random.normal(loc = s_ddisp_noerr,scale = s_ddisp_err) # Scramble data according to a normal distribution with 1 sigma = source_ddec_err [mas]

        # create Solver class
        class UnlensedModelPyMultiNest(Solver):
            """
            The lensed model, with a Gaussian likelihood.

            Args:
                data (:class:`numpy.ndarray`): an array containing the observed data
                abscissa (:class:`numpy.ndarray`): an array containing the points at which the data were taken
                modelfunc (function): a function defining the model
                sigma (float): the standard deviation of the noise in the data
                **kwargs: keyword arguments for the run method
            """

            # define the prior parameters
            
            # This is only for the absolute worst case scenario, where there is no best fit for the free model

            ra_min = -100000.
            ra_max = 100000.
            
            dec_min = -100000. 
            dec_max = 100000.
            
            pmra_min = -100000. # Bernard's star (fastest star in the sky) has a proper motion of ~10000 mas/yr
            pmra_max = 100000.
            
            pmdec_min = -100000.
            pmdec_max = 100000.
            
            dist_min = 1. # Proxima Centuri is 1.2948 pc away
            dist_max = 50000.

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
                self.raprime = cube[0]
                self.decprime = cube[1]
                self.pmraprime = cube[2]
                self.pmdecprime = cube[3]
                self.distprime = cube[4]
                
                params = cube.copy()
                
                # Put very wide Guassians around each best fit free parameter as our inverse transform sampling prior
                # Put very wide Guassians around each best fit free parameter as our inverse transform sampling prior
                params[0] = inverse_gaussian_cdf(self.raprime,y0,30.)
                params[1] = inverse_gaussian_cdf(self.decprime,y1,30.)
                params[2] = inverse_gaussian_cdf(self.pmraprime,y2,50.)
                params[3] = inverse_gaussian_cdf(self.pmdecprime,y3,50.)
                params[4] = np.absolute(inverse_gaussian_cdf(self.distprime,np.min([np.absolute(y4),4500]),5000.))

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
                ra = cube[0]
                dec = cube[1]
                pmra = cube[2]
                pmdec = cube[3]
                dist = cube[4]
                
                parms = np.array([ra,dec,pmra,pmdec,dist])

                # calculate the model
                ll = -analysis.free_2ll(self._data,self._sigma,s_ra0,s_dec0,parms)

                if np.isnan(ll)==True or np.isinf(ll)==True:
                    ll = -1.0e100
                
                return ll

        nlive = 700 # number of live points
        ndim = 5     # number of parameters
        tol = 0.1    # stopping criterion

        # run the algorithm
        solution = UnlensedModelPyMultiNest(data=s_ddisp, sigma=s_ddisp_err, n_dims=ndim,
                                                n_live_points=nlive, evidence_tolerance=tol,resume=False);

        logZpymnest = solution.logZ        # value of log Z
        logZerrpymnest = solution.logZerr  # estimate of the statistcal uncertainty on logZ

        rachain_pymnest = solution.samples[:,0] # extract chain of ra values
        decchain_pymnest = solution.samples[:,1] # extract chain if dec values
        pmrachain_pymnest = solution.samples[:,2] # extract chain of ra values
        pmdecchain_pymnest = solution.samples[:,3] # extract chain if dec values
        distchain_pymnest = solution.samples[:,4] # extract chain of ra values

        postsamples = np.vstack((rachain_pymnest, decchain_pymnest,pmrachain_pymnest,pmdecchain_pymnest,distchain_pymnest)).T

        # MULTINEST END

        y_bf = solution.samples[-1] # Save bf parameters

        # Go the rest of the way home!
        free_fit = scipy.optimize.minimize(lambda x: analysis.free_2ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
                                     x0=y_bf, # Specify the initial guess
                                     method = 'SLSQP', # Select Sequential Least SQuares Programming based minimizer
                                     tol= 1e-7, # Set the tolarance level for what is considered a minimum
                                     jac = '3-point', # Set the Jacobian to be of the 3-point type
                                     constraints=free_constraints, # Load parameter constraints specified by the constraints_fcns script
                                     options = {'maxiter':1000000}, # Set the miximum number of minimizer iterations
                                     )

        y_bf = free_fit.x # Extract best fit parameters
        free_2ll = free_fit.fun # Save the log likelihood of the free fit to the data

        # Save the best fit parameters and source id to a dictionary
        data_out = {'id':[s_id],'y0':[y_bf[0]],'y1':[y_bf[1]],'y2':[y_bf[2]],'y3':[y_bf[3]],'y4':[y_bf[4]],'2ll':[free_2ll]}

        # Convert the output dictionary to a pandas dataframe
        dataf = pd.DataFrame(data_out)
        
        if m==0: # Save w/ header if first row
            dataf.to_csv('./Results/FreeMultinest/free_'+file_id+'.csv', mode='a', index=False, header=(('s_id','delta_ra0 [mas]','delta_dec0 [mas]','pm_ra [mas/yr]','pm_dec[mas/yr]','dist [pc]','ts')))

        else: # Else save without header
            dataf.to_csv('./Results/FreeMultinest/free_'+file_id+'.csv', mode='a', index=False, header=False)

        np.savez('./Results/FreePostsamples/post_'+file_id+'.npz',postsamples=postsamples) # Save the postsamples

    else:
        continue
print('Analysis Complete')

import os
import sys
os.chdir('/home/mk7976/git/gaiablip') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

results_path = '/scratch/mk7976/fit_results/x1_new/free_multinest_results'
postsamples_path = '/scratch/mk7976/fit_results/x1_new/free_postsamples' # Specify the folder to save the postsamples to
catalog_path = '/scratch/mk7976/epoch_astrometry/lens_new' # Specify the folder to load the catalog from

import dynamics_fcns as df
import bh_prior_fcns
import scipy

priors = bh_prior_fcns.BH_priors()
dynamics = df.Dynamics()

# Change system path
os.chdir('/home/mk7976/git/gaiablip/analysis/') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import analysis_fcns as af
import pandas as pd

# import PyMultiNest Solver class
from pymultinest.solve import Solver

import constraint_fcns # Import the constraint functions script
free_constraints = constraint_fcns.cons_free # Tuple of functions contraining free model fitting

obs_info = pd.read_csv('./obs_info.csv', sep=",", header=None, skiprows = [0]) # Read observation info csv
obs_info.columns = ['t_obs','scan_angles'] # Specify what each column in obs_info file are
t_obs = obs_info['t_obs'].to_numpy() # Observation times [Julian years]
scan_angles = obs_info['scan_angles'].to_numpy() #  Scan angles [rad]
t_ref = 2017.5 # Reference time for calculating displacement [Julian years]
n_obs = len(t_obs) # Number of observations
dof = n_obs-5 # Degrees of freedom, number of data points minus number of parameters

blip_search = af.blip_search('bh')
catalog_list = os.listdir('/scratch/mk7976/epoch_astrometry/lens_new')

os.chdir('/home/mk7976/git/gaiablip/') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

# Specify the minimum free 2LL that a source must have to be saved
job_idx = int(sys.argv[1])
catalog_name = os.listdir(catalog_path)[job_idx] # Pick the file name corresponding to the job to be ananalyzed
file_number = catalog_name[18:31] # Pick the data file number from the data file name
catalog_info_name = 'gaia_info_'+file_number+'.csv' # Anitialize the name of the info file correpsponding to the data file

data = pd.read_pickle('/scratch/mk7976/epoch_astrometry/lens_new/'+catalog_name) 
misc_info_folder = '/scratch/ic2127/gaia_edr3_info/' # Specify the location of the folder containing the file with parallax and g magnitude data
misc_info_data = pd.read_csv(misc_info_folder+catalog_info_name) # Load in file containing parallax and g magnitude information

catalog_id_list = list(data['source_id']) # List of source IDs in the catalog

# Load random seed list
seed_info_folder = './analysis/seed_lists/'
seed_info = pd.read_csv('/scratch/mk7976/seed_lists/'+file_number+'_seeds.csv')

# Load the file with the results from the initial fit
results = pd.read_csv('/scratch/mk7976/fit_results/x1_new/free_fit_results/free_'+file_number+'.csv')

def inverse_gaussian_cdf(x,mu,sigma): # Define the inverse gaussian cdf function to be used for inverse transform sampling
    val = np.sqrt(2)*sigma*scipy.special.erfinv(2*x-1)+mu
    return val

for m in range(np.size(results['source_id'])):

    s_id = data.iat[m,0] # Source ID
    results_row = results.iloc[m] # Skip header
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

        s_ddisp_err = blip_search.disp_err(s_gmag)
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
                ll = -blip_search.free_2ll(self._data,self._sigma,s_ra0,s_dec0,parms)

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
        free_fit = scipy.optimize.minimize(lambda x: blip_search.free_2ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
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
            dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=(('s_id','delta_ra0 [mas]','delta_dec0 [mas]','pm_ra [mas/yr]','pm_dec[mas/yr]','dist [pc]','ts')))

        else: # Else save without header
            dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=False)

        np.savez(postsamples_path+'/post_'+file_number+'.npz',postsamples=postsamples) # Save the postsamples

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

        s_ddisp_err = blip_search.disp_err(s_gmag)
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
                ll = -blip_search.free_2ll(self._data,self._sigma,s_ra0,s_dec0,parms)

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
        free_fit = scipy.optimize.minimize(lambda x: blip_search.free_2ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
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
            dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=(('s_id','delta_ra0 [mas]','delta_dec0 [mas]','pm_ra [mas/yr]','pm_dec[mas/yr]','dist [pc]','ts')))

        else: # Else save without header
            dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=False)

        np.savez(postsamples_path+'/post_'+file_number+'.npz',postsamples=postsamples) # Save the postsamples

    else:

        # Save the best fit parameters and source id to a dictionary
        data_out = {'id':[s_id],'y0':[y[0]],'y1':[y[1]],'y2':[y[2]],'y3':[y[3]],'y4':[y[4]],'2ll':[chisq]}

        # Convert the output dictionary to a pandas dataframe
        dataf = pd.DataFrame(data_out)
        

        if m==0: # Save w/ header if first row
            dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=(('s_id','delta_ra0 [mas]','delta_dec0 [mas]','pm_ra [mas/yr]','pm_dec[mas/yr]','dist [pc]','ts')))

        else: # Else save without header
            dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=False)

print('Analysis Complete')

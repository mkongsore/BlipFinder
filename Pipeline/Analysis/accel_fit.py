# GAIA Accel Model Script
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

# Initialize constraint functions
free_constraints = constraint_fcns.cons_free # Tuple of functions contraining free model fitting

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

for n in range(np.size(free_multinest_results['-2ll'])):   
    s_id = free_multinest_results.iat[n,0]
    results_row = free_multinest_results.iloc[n]

    s_free_ra0 = float(results_row[1])
    s_free_dec0 = float(results_row[2])
    s_free_pmra = float(results_row[3])
    s_free_pmdec = float(results_row[4])
    s_free_dist = float(results_row[5])
    free_ll = float(results_row[6])

    if free_ll>152.: # Do lensed fit if greater than five sigma chi_sq

        s_idx = catalog_id_list.index(int(s_id))
        
        s_row = data.iloc[s_idx] # Pick particular RA row corresponding to source of interest
        s_info_row = source_info.iloc[s_idx] # Pick the particular data file row corresponding to the source of interest

        s_ra0 = float(s_row[1]) # RA of source at first observation epoch [deg]
        s_dec0 = float(s_row[2]) # DEC of source at first observation epoch [deg]
        s_ddisp_noerr = np.array(s_row[3:]) # Change in RA [mas]
        s_dist = float(s_info_row[3]) # Estimated distance to source [pc]
        s_gmag = float(s_info_row[6]) # G magntiude of source

        if np.isnan(s_gmag)==True:
            s_gmag = 19.571021 # Set G magntiude to the median of the catalog if it is not available

        np.random.seed(int(seed_info.iat[n,2])) # Set a random seed to ensure data gets scrambled in the same way every time

        s_ddisp_err = analysis.disp_err(s_gmag)
        s_ddisp = s_ddisp_noerr
        s_ddisp = np.random.normal(loc = s_ddisp_noerr,scale = s_ddisp_err) # Scramble data according to a normal distribution with 1 sigma = source_ddec_err [mas]

        guess_ra_accel = 0.
        guess_dec_accel = 0.

        accel_fit_guess = np.array([s_free_ra0,s_free_dec0,s_free_pmra,s_free_pmdec,s_free_dist,guess_ra_accel,guess_dec_accel])

        # Run lensed fit
        accel_fit = scipy.optimize.minimize(lambda x : analysis.free_7p_ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
                x0=accel_fit_guess, # Specify the initial guess
                method='SLSQP', # Select Sequential Least SQuares Programming based minimizer
                tol=1e-5, # Set the tolerence for what is classified as a sucessful fit
                jac = '3-point', # Set the tolarance level for what is considered a minimum
                constraints=free_constraints, # Load the parameter constraints specified by the comnstraints_fcns script
                options={'maxiter':100000})

        bf_ts = accel_fit.fun
        bf_x = accel_fit.x # Save the best fit parameters obtained from the fit

        ################
        # Save Results #
        ################

        # Save the best fit parameters and source id to a dictionary
        data_out = {'id':[s_id],'x0':[bf_x[0]],'x1':[bf_x[1]],
                    'x2':[bf_x[2]],'x3':[bf_x[3]],'x4':[bf_x[4]],
                    'x5':[bf_x[5]],'x6':[bf_x[6]],'ts':[bf_ts]}

        # Convert the output dictionary to a pandas dataframe
        dataf = pd.DataFrame(data_out)

        # Save the output to a csv file corresponding to the datafile the source is in to scratch

        if first_sig_event==True: # Save w/ header if first >5sigma source
            dataf.to_csv('./Results/AccelScipy/accel_'+str(file_id)+'.csv', mode='a', index=False, header=(('s_id','s_free_delta_ra0 [mas]','s_free_delta_dec0 [mas]','s_free_pm_ra [mas/yr]','s_free_pm_dec[mas/yr]','s_free_dist [pc]','s_accel_ra [mas/yr/yr]','s_accel_dec [mas/yr/yr]','-2ll')))
            first_sig_event = False

        else: # Else append without header
            # Save the output to a csv file corresponding to the datafile the source is in to scratch
            dataf.to_csv('./Results/AccelScipy/accel_'+str(file_id)+'.csv', mode='a', index=False, header=False)

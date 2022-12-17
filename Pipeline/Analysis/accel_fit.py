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

results_path = '/scratch/mk7976/fit_results/x1_new/accel_fit_results'
catalog_path = '/scratch/mk7976/epoch_astrometry/lens_new' # Specify the folder to load the catalog from

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

from scipy import optimize as spo

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

os.chdir('/home/mk7976/git/gaiablip/') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

# Specify the minimum unlensed 2LL that a source must have to be saved
job_idx = int(sys.argv[1])

catalog_name = os.listdir(catalog_path)[job_idx] # Pick the f`ile name corresponding to the job to be ananalyzed
file_number = catalog_name[18:31] # Pick the data file number from the data file name
catalog_info_name = 'gaia_info_'+file_number+'.csv' # Anitialize the name of the info file correpsponding to the data file

data = pd.read_pickle('/scratch/mk7976/epoch_astrometry/lens_new/'+catalog_name) 
catalog_id_list = list(data['source_id'])

misc_info_folder = '/scratch/ic2127/gaia_edr3_info/' # Specify the location of the folder containing the file with parallax and g magnitude data
misc_info_data = pd.read_csv(misc_info_folder+catalog_info_name) # Load in file containing parallax and g magnitude information

results = pd.read_csv('/scratch/mk7976/fit_results/x1_new/free_multinest_results/free_'+file_number+'.csv')

# Load random seed list 
seed_info_folder = './analysis/seed_lists/'
seed_info = pd.read_csv('/scratch/mk7976/seed_lists/'+file_number+'_seeds.csv')

first_sig_event = True

for n in range(np.size(results['ts'])):   
    s_id = results.iat[n,0]
    results_row = results.iloc[n]

    s_free_ra0 = float(results_row[1])
    s_free_dec0 = float(results_row[2])
    s_free_pmra = float(results_row[3])
    s_free_pmdec = float(results_row[4])
    s_free_dist = float(results_row[5])
    free_ll = float(results_row[6])

    if free_ll>152.: # Do lensed fit if greater than five sigma chi_sq

        s_idx = catalog_id_list.index(int(s_id))
        
        s_row = data.iloc[s_idx] # Pick particular RA row corresponding to source of interest
        s_info_row = misc_info_data.iloc[s_idx] # Pick the particular data file row corresponding to the source of interest

        s_ra0 = float(s_row[1]) # RA of source at first observation epoch [deg]
        s_dec0 = float(s_row[2]) # DEC of source at first observation epoch [deg]
        s_ddisp_noerr = np.array(s_row[3:]) # Change in RA [mas]
        s_dist = float(s_info_row[3]) # Estimated distance to source [pc]
        s_gmag = float(s_info_row[6]) # G magntiude of source

        if np.isnan(s_gmag)==True:
            s_gmag = 19.571021 # Set G magntiude to the median of the catalog if it is not available

        np.random.seed(int(seed_info.iat[n,2])) # Set a random seed to ensure data gets scrambled in the same way every time

        s_ddisp_err = blip_search.disp_err(s_gmag)
        s_ddisp = s_ddisp_noerr
        s_ddisp = np.random.normal(loc = s_ddisp_noerr,scale = s_ddisp_err) # Scramble data according to a normal distribution with 1 sigma = source_ddec_err [mas]

        guess_ra_accel = 0.
        guess_dec_accel = 0.

        accel_fit_guess = np.array([s_free_ra0,s_free_dec0,s_free_pmra,s_free_pmdec,s_free_dist,guess_ra_accel,guess_dec_accel])

        # Run lensed fit
        accel_fit = spo.minimize(lambda x : blip_search.free_7p_ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
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
            dataf.to_csv(results_path+'/accel_'+str(file_number)+'.csv', mode='a', index=False, header=(('s_id','s_free_delta_ra0 [mas]','s_free_delta_dec0 [mas]','s_free_pm_ra [mas/yr]','s_free_pm_dec[mas/yr]','s_free_dist [pc]','s_accel_ra [mas/yr/yr]','s_accel_dec [mas/yr/yr]','ts')))
            first_sig_event = False

        else: # Else append without header
            # Save the output to a csv file corresponding to the datafile the source is in to scratch
            dataf.to_csv(results_path+'/accel_'+str(file_number)+'.csv', mode='a', index=False, header=False)

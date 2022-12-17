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

results_path = '/scratch/mk7976/fit_results/x1/blip_fit_results'
catalog_path = '/scratch/mk7976/epoch_astrometry/lens/x1' # Specify the folder to load the catalog from

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
blip_constraints = constraint_fcns.cons_blip # Tuple of functions contraining blip model fitting

obs_info = pd.read_csv('./obs_info.csv', sep=",", header=None,skiprows=[0]) # Read observation info csv
obs_info.columns = ['t_obs','scan_angles'] # Specify what each column in obs_info file are
# Extrapolate info from obs_info file
t_obs = obs_info['t_obs'].to_numpy() # Observation times [Julian years]
scan_angles = obs_info['scan_angles'].to_numpy() #  Scan angles [rad]
t_ref = 2017.5 # Reference time for calculating displacement [Julian years]
n_obs = len(t_obs) # Number of observations
dof = n_obs-5

blip_search = af.blip_search('bh')
catalog_list = os.listdir('/scratch/mk7976/epoch_astrometry/lens/x1')

os.chdir('/home/mk7976/git/gaiablip/') # Go to the parents directory
cwd = os.getcwd() # Retrieve directory of current running processes
sys.path.insert(0, cwd) # Change the system path to the current running directory

# Specify the minimum unlensed 2LL that a source must have to be saved
job_idx = int(sys.argv[1])

catalog_name = os.listdir(catalog_path)[job_idx] # Pick the file name corresponding to the job to be ananalyzed
file_number = catalog_name[21:34] # Pick the data file number from the data file name
catalog_info_name = 'gaia_info_'+file_number+'.csv' # Anitialize the name of the info file correpsponding to the data file

data = pd.read_pickle('/scratch/mk7976/epoch_astrometry/lens/x1/'+catalog_name) 
catalog_id_list = list(data['source_id'])

misc_info_folder = '/scratch/ic2127/gaia_edr3_info/' # Specify the location of the folder containing the file with parallax and g magnitude data
misc_info_data = pd.read_csv(misc_info_folder+catalog_info_name) # Load in file containing parallax and g magnitude information

results = pd.read_csv('/scratch/mk7976/fit_results/x1/free_multinest_results/free_'+file_number+'.csv')

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

        guess_lens_pm_ra = -1/((t_obs[-1]-t_obs[0])/2.)*(s_ddisp[0]/np.cos(scan_angles[0]) +
                    s_ddisp[-1]/np.cos(scan_angles[-1]))/(np.tan(scan_angles[0]) - np.tan(scan_angles[-1])) # Source RA proper motion guess [mas/yr]
        guess_lens_pm_dec = -1/((t_obs[-1]-t_obs[0])/2.)*(s_ddisp[0]/np.sin(scan_angles[0]) +
                    s_ddisp[-1]/np.sin(scan_angles[-1]))/(1/np.tan(scan_angles[0]) - 1/np.tan(scan_angles[-1])) # Source DEC proper motion guess [mas/yr]

        guess_ra_0_l = 0.1 # Relative starting position of source [mas]
        guess_dec_0_l = 0.1 # [mas]
        guess_lens_dist = np.min([6347.682620000001,s_free_dist/2.]) # Guess distance to source (median of catalog) [pc]
        guess_lens_m = 7. # Guess lens mass [sm]

        lens_fit_guess = np.array([s_free_ra0,s_free_dec0,s_free_pmra,s_free_pmdec,s_free_dist,guess_ra_0_l,guess_dec_0_l,guess_lens_pm_ra,guess_lens_pm_dec,guess_lens_dist,guess_lens_m])

        # Run lensed fit
        lens_prior_fit = spo.minimize(lambda x : blip_search.ts_priors(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
                x0=lens_fit_guess, # Specify the initial guess
                method='SLSQP', # Select Sequential Least SQuares Programming based minimizer
                tol=1e-5, # Set the tolerence for what is classified as a sucessful fit
                jac = '3-point', # Set the tolarance level for what is considered a minimum
                constraints=blip_constraints, # Load the parameter constraints specified by the comnstraints_fcns script
                options={'maxiter':100000})

        bf_ts = blip_search.llr(s_ddisp,s_ddisp_err,free_ll,s_ra0,s_dec0,lens_prior_fit.x) # Save the test statistic obtained from the fit
        bf_x = lens_prior_fit.x # Save the best fit parameters obtained from the fit

        ################
        # Save Results #
        ################

        # Save the best fit parameters and source id to a dictionary
        data_out = {'id':[s_id],'x0':[bf_x[0]],'x1':[bf_x[1]],'x2':[bf_x[2]],'x3':[bf_x[3]],'x4':[bf_x[4]],
                    'x5':[bf_x[5]],'x6':[bf_x[6]],'x7':[bf_x[7]],'x8':[bf_x[8]],'x9':[bf_x[9]],'x10':[bf_x[10]],'ts':[bf_ts]}

        # Convert the output dictionary to a pandas dataframe
        dataf = pd.DataFrame(data_out)

        # Save the output to a csv file corresponding to the datafile the source is in to scratch

        if first_sig_event==True: # Save w/ header if first >5sigma source
            dataf.to_csv(results_path+'/blip_'+str(file_number)+'.csv', mode='a', index=False, header=(('s_id','s_delta_ra0 [mas]','s_delta_dec0 [mas]','s_pm_ra [mas/yr]','s_pm_dec[mas/yr]','s_dist [pc]','l_delta_ra0 [mas]','l_delta_dec0 [mas]','l_pm_ra [mas/yr]','l_pm_dec[mas/yr]','l_dist [pc]','l_mass [sm]','ts')))
            first_sig_event = False

        else: # Else append without header
            # Save the output to a csv file corresponding to the datafile the source is in to scratch
            dataf.to_csv(results_path+'/blip_'+str(file_number)+'.csv', mode='a', index=False, header=False)

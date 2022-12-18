# GAIA Free Trajectory Analysis Script
# I-Kai Chen, Marius Kongsore, Ken Van Tilburg
# December 2022

# This file as part of a larger GitHub package
# https://github.com/mkongsore/BlipFinder

############################
# Load Functions and Files #
############################
 
# File paths
data_path = '../Data' # Specify folder to load data from
source_info_path = '../SourceInfo' # Specift path to read EDR3 source info from
results_path = '../Results/FreeScipy' # Specify path to save results to

# Import system packages
import sys
import os

# Import public packages
import scipy as sp
import pandas as pd
import numpy as np

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
t_obs = obs_info['t_obs [Julian Year]'].to_numpy() # Observation times [Julian years]
scan_angles = obs_info['Scan Angles [rad]'].to_numpy() #  Scan angles [rad]
t_ref = 2017.5 # Reference time for calculating displacement [Julian years]

file_name = os.listdir(catalog_path)[job_idx] # Pick the file name corresponding to the job to be analyzed
file_id = str(file_name[11:24]) # Pick the data file number from the data file name
info_file_name = 'gaia_info_'+file_id+'.csv' # Initialize the name of the info file correpsponding to the data file


# Load data file containing the displacement-time coordinates
data = data_path+'/gaia_epoch_'+file_id+'.pkl' # Specify the location of the obs files
data = pd.read_pickle(data) # Load observation data from pickle file
source_info = pd.read_csv(source_info_path+info_file_name) # Load in file containing parallax and g magnitude information

# Load random seed list
seed_info = pd.read_csv('./Analysis/mock_error_seeds.csv')

print('Files Loaded')

###########################
# RUN FREE MODEL ANALYSIS #
###########################

print('Analysis Loop Commenced')

# Go through data file rows one-by-one, corresponding to analyzing one source data set at a time
for n in np.arange(0,np.shape(data)[0]): # n is the row number

    s_data_row = data.iloc[n] # Pick particular catalog  row corresponding to source of interest
    s_info_row = misc_info_data.iloc[n] # Pick the particular data file row corresponding to the source of interest

    s_id = data.iat[n,0] # Specify the source ID
    s_ra0 = float(s_data_row[1]) # RA of source at first observation epoch [deg]
    s_dec0 = float(s_data_row[2]) # DEC of source at first observation epoch [deg]
    s_ddisp_noerr = np.array(s_data_row[3:]) # Change in displacement [mas]
    s_dist = float(s_info_row[3]) # Estimated distance to source [pc]
    s_gmag = float(s_info_row[6]) # G magntiude of source

    if np.isnan(s_gmag)==True:
        s_gmag = 19.571021 # Set G magntiude to the median of the catalog if it is not available

    np.random.seed(int(seed_info.iat[n,2])) # Set a random seed to ensure data gets scrambled in the same way every time, based on the source distance
    s_ddisp_err = analysis.disp_err(s_gmag) #  Compute the error of each data point [mas]
    s_ddisp = np.random.normal(loc = s_ddisp_noerr,scale = s_ddisp_err) # Scramble data according to a normal distribution with 1 sigma = source_ddisp_err [mas]

    # Now guess initial fit parameters

    # Source RA proper motion guess [mas/yr]
    s_free_guess_pm_ra = -1/((t_obs[-1]-t_obs[0])/2.)*(s_ddisp[0]/np.cos(scan_angles[0]) +
                s_ddisp[-1]/np.cos(scan_angles[-1]))/(np.tan(scan_angles[0]) - np.tan(scan_angles[-1]))
    # Source DEC proper motion guess [mas/yr]
    s_free_guess_pm_dec = -1/((t_obs[-1]-t_obs[0])/2.)*(s_ddisp[0]/np.sin(scan_angles[0]) +
                s_ddisp[-1]/np.sin(scan_angles[-1]))/(1/np.tan(scan_angles[0]) - 1/np.tan(scan_angles[-1])) # Source DEC proper motion guess [mas/yr]

    s_free_guess_ra0 = 0. # Relative starting ra guess [mas]
    s_free_guess_dec0 = 0. # Relative starting dec guess [mas]
    s_free_guess_dist = 6347.68262 # Distance to source guess (median of catalog) [pc]

    # Create array of guesses
    s_free_guess = np.array([s_free_guess_ra0,s_free_guess_dec0,
        s_free_guess_pm_ra,s_free_guess_pm_dec,s_free_guess_dist]) # Construct starting point coordinate array
    
    # Run the free model fit, using the scipy.optimize minimizer and the SLSQP method
    free_fit = sp.optimize.minimize(lambda x: analysis.free_2ll(s_ddisp,s_ddisp_err,s_ra0,s_dec0,x),
                                     x0=s_free_guess, # Specify the initial guess
                                     method = 'SLSQP', # Select Sequential Least SQuares Programming based minimizer
                                     tol= 1e-7, # Set the tolarance level for what is considered a minimum
                                     jac = '3-point', # Set the Jacobian to be of the 3-point type
                                     constraints=free_constraints, # Load parameter constraints specified by the constraints_fcns script
                                     options = {'maxiter':1000000}, # Set the miximum number of minimizer iterations
                                     )

    # Save the best fit parameters and source id to a dictionary
    data_out = {'id':[s_id],'y0':[free_fit.x[0]],'y1':[free_fit.x[1]],'y2':[free_fit.x[2]],'y3':[free_fit.x[3]],'y4':[free_fit.x[4]],'-2ll':[free_fit.fun]}

    # Convert the output dictionary to a pandas dataframe
    dataf = pd.DataFrame(data_out)

    # Save the output to a csv file corresponding to the datafile the source is in to scratch

    if n==0: # Save w/ header if first row
        dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=(('source_id','delta_ra0 [mas]','delta_dec0 [mas]','pm_ra [mas/yr]','pm_dec[mas/yr]','dist [pc]','ts')))
    else: # Else save without header
        dataf.to_csv(results_path+'/free_'+file_number+'.csv', mode='a', index=False, header=False)

print('Free Fit Complete')

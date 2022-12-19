import numpy as np
import glob
import pandas as pd
import time
import scipy.constants as const
import sys
sys.path.append("../")
import bh_prior_fcns
import dynamics_fcns


row = int(sys.argv[1]) # row of Gaia file (0 - 3385)

pix = pd.read_csv('pixel/BH_number.csv', skiprows = range(1,row + 1),nrows = 1)
start = str(pix['start'][0]).zfill(6)
end = str(pix['end'][0]).zfill(6)

dyn = dynamics_fcns.Dynamics()
t_ar = dyn.t_obs
angle = dyn.scan_angles
t0 = dyn.t_ref


# Read in Gaia information
gaia = pd.read_csv(f'../Sourceinfo/gaia_info_{start}-{end}.csv')


# create null trajectory
df = pd.DataFrame()
df['source_id'] = gaia['source_id']
df['ra'] = gaia['ra']
df['dec'] = gaia['dec']
df[t_ar] = 0

for i in range(len(gaia)):
    traj = dyn.unlens_AL(gaia['ra'][i], gaia['dec'][i], gaia['pmra'][i],
                         gaia['pmdec'][i], gaia['distance'][i])
    df.loc[i, t_ar] = traj

df.to_pickle(f'null/gaia_epoch_{start}-{end}.pkl')

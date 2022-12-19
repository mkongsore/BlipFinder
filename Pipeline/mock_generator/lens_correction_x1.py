import numpy as np
import pandas as pd
import scipy.constants as const
import sys
sys.path.append("/home/ic2127/gaiablip")

import dynamics_fcns



row = int(sys.argv[1]) # row of Gaia file (0 - 3385)

seed = 0
np.random.seed(seed) # set random seed


pix = pd.read_csv('pixel/pixel_list.csv', skiprows = range(1,row + 1),nrows = 1)
start = str(pix['start'][0]).zfill(6)
end = str(pix['end'][0]).zfill(6)

dyn = dynamics_fcns.Dynamics()
t_ar = dyn.t_obs
angle = dyn.scan_angles
t0 = dyn.t_ref
n_obs = dyn.n_obs


# Read in Gaia information
gaia = pd.read_csv(f'../Sourceinfo/gaia_info_{start}-{end}.csv')

# Read in unlens trajectory
unlens_traj = pd.read_pickle(f'null/gaia_epoch_{start}-{end}.pkl')

# Read in lens information
lens = pd.read_csv(f'lens_info/lens_info_{start}-{end}.csv')


# calculate lensing correction
correction = np.zeros((len(unlens_traj), n_obs))
for i in range(len(lens)):
    for j in range(7, len(lens.loc[0])):
        if lens.iloc[i][j] == -1 or np.isnan(lens.iloc[i][j]):
            break
        else:
            l_dist = lens['distance'][i]*1000
            ind_g = np.where(gaia['source_id'] == lens.iloc[i][j])[0][0]
            ind_t = np.where(unlens_traj['source_id'] == lens.iloc[i][j])[0][0]

            assert ind_g == ind_t, 'indice not match'

            if gaia['distance'][ind_g] > l_dist:

                ra_s = gaia['ra'][ind_g]
                dec_s = gaia['dec'][ind_g]
                ind = ind_g


                traj_lens = dyn.lensed_AL(ra_s, dec_s, gaia['pmra'][ind], gaia['pmdec'][ind], gaia['distance'][ind],
                                     lens['ra'][i], lens['dec'][i], lens['pmra'][i], lens['pmdec'][i], l_dist, lens['mass'][i])

                free = unlens_traj.iloc[ind_t, range(3, n_obs+3)]

                correction[ind_t] = correction[ind_t] + (traj_lens - free)


df_lensed = unlens_traj.copy()
df_lensed.iloc[:, range(3, n_obs + 3)] = df_lensed.iloc[:, range(3, n_obs + 3)] + correction



df_lensed.to_pickle(f'lensed_traj/gaia_epoch_lensed_{start}-{end}_new.pkl')

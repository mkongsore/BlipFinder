import healpy as hp
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import scipy.constants as const
import sys
sys.path.append("../")
import bh_prior_fcns
import rotation as rot
import time
from dynamics_fcns import _eins_r



row = int(sys.argv[1]) # row of Gaia file (0 - 3385)

seed = 0
np.random.seed(seed) # set random seed



def _random_loc(n):
    """
    Return random location in healpix number n at level 8

    Parameters
    ----------
    n : int/np.array
        healpix number with NSIDE 2**8

    Returns
    -------
    The 2D array of (ra, dec)
    """
    size = len(n)
    for i in range(8,29):
        n = n*4 + np.random.randint(0,4, size = size)
    ra, dec = hp.pix2ang(2**29, n, nest = True, lonlat = True)
    rand_loc = np.random.rand(size,2)*1e-7 - 5e-8
    rot = np.array([[1,-1],
                [1, 1]])/np.sqrt(2)
    rand_loc = rand_loc.dot(rot)
    dec = dec + rand_loc[:,1]
    ra = ra + rand_loc[:,0]/np.cos(dec*const.degree)
    return ra, dec



def _perp_pm(v, l, b, sun_rot = np.array([0, 220, 0]),
            sun_pec = np.array([11.1, 12.24, 7.25])):
    """
    Returns the projection of velocity onto the celestial sphere in
    Cartstian coordinate.
    x-axis: points towards the Galactic center.
    y-axis: points towards the direction of rotation.
    z-axis: points towards the Galactic north pole.

    Parameters
    ----------
    v : np.array
        velocity in Cartesian coordinate in staionary frame.
    l : float
        galactic longtitude in degrees.
    b : float
        galactic lattitude in degrees.
    sun_rot : np.array
        Sun's rotational velocity in Cartesian coordinate (default: [0, 220, 0]).
    sun_pec : np.array
        Sun's peculiar velocity in Cartesian coordinate (default: [11.1, 12.24, 7.25]).

    Returns
    -------
    The projected velocity (vx, vy, vz)
    """
    l = l*const.degree
    b = b*const.degree

    projection = np.array([np.cos(b)*np.cos(l),
                           np.cos(b)*np.sin(l),
                           np.sin(b)]).reshape((1,3))
    v = v - sun_rot - sun_pec
    perp = v - (projection.dot(v.T).T)*projection
    return perp[0]


prior = bh_prior_fcns.BH_priors()

pix = pd.read_csv('pixel/BH_number.csv', skiprows = range(1,row + 1),nrows = 1)
start = pix['start'][0]
end = pix['end'][0]


# number of lens
N = np.random.poisson(pix['BH_num'][0])#*norm)

# drawing BH distance with linear interpolation
r_ar = np.array(pix.columns[5:], dtype = float)
f_dist = interp1d(pix[r_ar.astype(str)].to_numpy()[0], r_ar)
lens_dist = f_dist(np.random.rand(N))#*norm)


# drawing BH mass with linear interpolation
mass_cdf = pd.read_csv('cdf/BH_mass_cdf.csv')
f_mass = interp1d(mass_cdf['cdf'], mass_cdf['mass'])
mass = f_mass(np.random.rand(N))


# drawing BH location
loc = np.random.choice(np.arange(start, end + 1), N)
lens_ra, lens_dec = _random_loc(loc)


# coordinate transformation from equatorial to galactic
r_C2G = hp.rotator.Rotator(coord = ('C', 'G'))
b, l = r_C2G((90 - lens_dec)*const.degree, lens_ra*const.degree)/const.degree
b = 90 - b


# drawing BH velocity in cylindrical coordinate
v_rot = np.array([0, -220,0])
v_cyl = np.random.multivariate_normal(np.zeros(3),
                                      np.array([[35,  0,  0],
                                                [ 0, 25,  0],
                                                [ 0,  0, 20]])**2, size = N)


# rotate velocity into cartesian coordinate
v_cart = np.zeros((N,3))
for i in range(N):
    theta = prior._theta_cyl(lens_dist[i], l[i], b[i])
    v_cart[i] = (rot.rot_cyl2cart(theta)).dot(v_rot + v_cyl[i])


# drawing BH natal kick
cos_theta = 1 - 2*np.random.rand(N)
sin_theta = np.sqrt(1 - cos_theta**2)
phi = np.random.rand(N)*2*np.pi

natal_cdf = pd.read_csv('cdf/natal_cdf.csv')
f_natal = interp1d(natal_cdf['cdf'], natal_cdf['natal'])
kick = f_natal(np.random.rand(N))
v_cart[:,2] += kick*cos_theta
v_cart[:,0] += kick*sin_theta*np.cos(phi)
v_cart[:,1] += kick*sin_theta*np.sin(phi)


# projecting BH velocity onto celestial sphere
pm = np.zeros((N,2))
for i in range(N):
    v_perp = _perp_pm(v_cart[i], l[i], b[i])
    v_lb = (rot.rot_cart2galactic(l[i], b[i])).dot(v_perp)[1:]
    pm[i] = (rot.rot_vec_radec2lb(l[i], b[i]).T).dot(v_lb)

pm[:,0] = pm[:,0]/lens_dist/const.parsec*const.Julian_year/const.arcsec*1000
pm[:,1] = pm[:,1]/lens_dist/const.parsec*const.Julian_year/const.arcsec*1000


gaia_info = pd.read_csv(f'/scratch/ic2127/gaia_edr3_info/gaia_info_{str(start).zfill(6)}-{str(end).zfill(6)}.csv')


hp_start_ind = np.zeros((end - start + 1), dtype = int)
hp_end_ind = np.zeros((end - start + 1), dtype = int)

source_pix = (gaia_info['source_id']//2**(59 - 16)).to_numpy()
diff1 = source_pix[1:] - source_pix[:-1]
hp_end_ind = np.where(diff1==1)[0] + 1
hp_start_ind = np.insert(hp_end_ind, 0, 0)
hp_end_ind = np.append(hp_end_ind, len(gaia_info))



# record close encounters with stars with minimum maximum deflection
# at least 5 micro arcsec, and closest approaching between t (-10, 10)
encounter = []

for i in range(N):
    try:
        hp_ind = hp.ang2pix(2**8, lens_ra[i], lens_dec[i], nest = True, lonlat = True) - start
        gaia_hp = gaia_info.loc[range(hp_start_ind[hp_ind],hp_end_ind[hp_ind])]
    except:
        gaia_hp = gaia_info
    behind = gaia_hp.loc[gaia_hp['distance']/(lens_dist[i]*1000)>1]
    d_ra = (lens_ra[i] - behind['ra'])*np.cos(lens_dec[i]*const.degree)*3600*1000
    d_dec = (lens_dec[i] - behind['dec'])*3600*1000

    v_ra = pm[i,0] - behind['pmra']
    v_dec = pm[i,1] - behind['pmdec']

    v_sq = v_ra**2 + v_dec**2

    t_min = -(v_ra*d_ra + v_dec*d_dec)/v_sq

    b_min = np.sqrt((v_ra*d_dec - v_dec*d_ra)**2/v_sq)
    b_eins = _eins_r(mass[i], behind['distance'], lens_dist[i]*1000)
    u = b_min/b_eins
    u[u<np.sqrt(2)] = np.sqrt(2)
    d_max = b_eins*u/(u**2 + 2)
    enc = (behind['source_id'].loc[(np.abs(t_min)<10) & d_max > 5e-3]).to_numpy()
    encounter.append(enc)


# save the lens information
df = pd.DataFrame()
df['lens_id'] = range(N)
df['mass'] = mass
df['ra'] = lens_ra
df['dec'] = lens_dec
df['distance'] = lens_dist
df['pmra'] = pm[:,0]
df['pmdec'] = pm[:,1]


max_en = 0
for i in encounter:
    if len(i)> max_en:
        max_en = len(i)

for i in range(max_en):
    df['en'+str(i+1)] = -1


for i in range(len(encounter)):
    for j in range(len(encounter[i])):
            df['en'+str(j + 1)][i] = encounter[i][j]

df = df.loc[df['en1'] != -1]
df = df.reset_index(drop = True)
df['lens_id'] = range(len(df))

#print('file ', time.time() - start1)

df.to_csv(f'lens_info/lens_info_{str(start).zfill(6)}-{str(end).zfill(6)}.csv',
          index = False)

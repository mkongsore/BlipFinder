import numpy as np

### Constraints

def positive_source_distance(x):
    "Require that the distance to the source be positive"
    return x[4]-1. # Require d_source > 1. [pc]

def positive_lens_distance(x):
    "Require that the distance to the source be positive"
    return x[9]-1. # Require d_source > 1. [pc]

def source_behind_lens(x):
    "Require that the source is further away than the lens"
    return x[4] - x[9] - 1. # Require d_source > d_lens+1. [pc]

def lens_mass_positive(x):
    return x[10]

def lens_mass_above_lb(x):
    "Require that the mass of the lens is above the lower bound of the bh mass distribution"
    return x[10]-4.60 # Require m_lens > 4.59 [sm]

def lens_mass_below_ub(x):
    "Require that the mass of the lens is below the upper bound of the bh mass distribution"
    return -x[10] + 86.22 # Require m_lens < 86.22 [sm]

def blip_event(x):
    "Require that the event being fitted to is a blip"
    pm_ra = x[2] - x[7] # Relative RA proper motion for source and lens [mas/yr]
    pm_dec = x[3] - x[8] # Relative DEC proper motion for source and lens [mas/yr]
    mission_duration = 5.0 # [yr]
    return (pm_ra**2 + pm_dec**2)*mission_duration**2 - ((x[6]-x[1])*pm_ra - (x[5]-x[0])*pm_dec)**2/(pm_ra**2 + pm_dec**2) # Require that the a blip is being fitted.

def source_ra_ub(x):
    return 648000 - x[0] # [mas]

def source_dec_ub(x):
    return 648000 - x[1] # [mas]

def source_ra_lb(x):
    return 648000 + x[0] # [mas]

def source_dec_lb(x):
    return 648000 + x[1] # [mas]

def lens_ra_ub(x):
    return 648000 - x[5] # [mas]

def lens_dec_ub(x):
    return 648000 - x[6] # [mas]

def lens_ra_lb(x):
    return 648000 + x[5] # [mas]

def lens_dec_lb(x):
    return 648000 + x[6] # [mas]

cons_blip = ({'type':'ineq', 'fun':positive_lens_distance},
        {'type':'ineq', 'fun':positive_source_distance},
        {'type':'ineq', 'fun':source_behind_lens},
        {'type':'ineq', 'fun':lens_mass_above_lb},
        {'type':'ineq', 'fun':lens_mass_below_ub},
        {'type':'ineq', 'fun':blip_event})

cons_free = ({'type':'ineq', 'fun':positive_source_distance})

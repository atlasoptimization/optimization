"""
The goal of this script is to compare the ambiguity resolution function to a 
guess of a distance employing a global solver directly for the distance d
For this, do the following:
    1. Definitions and imports
    2. Simulate data
    3. Solve the estimation problem 
    4. Compare results from both methods

"""

"""
    1. Definitions and imports -----------------------------------------------
"""



# i) Imports

import sys

sys.path.insert(0, './AR_compilation')

import Support_funs_AR as sf
import numpy as np
import Ambiguity_resolution as AR
import time
from skopt import gp_minimize


# ii) Basic definitions

n_obs=5
distance_true=5

weights=np.ones([1])
wavelengths=np.linspace(0.01,0.05,n_obs)
phase_variances=np.ones([n_obs])*0.01


# iii) Objective function

def residuals(observations, weights,distances,wavelengths):
    
    observations_pred, _=sf.Generate_data(weights,distances,wavelengths)
    phi_pred=np.angle(observations_pred)
    phi_diff=np.angle(np.conj(observations)*observations_pred)
    
    return phi_diff, phi_pred

def obj_fun(observations, weights,distances,wavelengths):
    phi_residuals,_ = residuals(observations, weights,distances,wavelengths)
    norm_resid=np.linalg.norm(phi_residuals,1)
    return norm_resid


# iv) Initialize loop

n_trials=10
d_residuals_AR=np.zeros(n_trials)
d_residuals_global=np.zeros(n_trials)




"""
    2. Simulate data ---------------------------------------------------------
"""

for k in range(n_trials):
    
    
    # i) Generate the data
    
    d_true=np.random.uniform(0,10)
    distances=d_true*np.ones([1])
    
    observations, _=sf.Generate_data(weights,distances,wavelengths)
    observations_noisy, _=sf.Generate_data_noisy(weights,distances,wavelengths,phase_variances)
    
    
    
    
    """
        3. Solve the estimation problem ------------------------------------------
    """
    
    
    # i) Solve the problem with our algorithm
    
    cons=['d_opt<=20']
    optim_opts=sf.Setup_optim_options(n_obs, constraints=cons)
    
    d_AR,N_AR,r_AR=AR.Ambiguity_resolution(observations, wavelengths, phase_variances,optim_opts)
    #d_noise,N_noise,r_noise=AR.Ambiguity_resolution(observations, wavelengths, phase_variances,optim_opts)
    
    
    # ii) Solve the problem with a global algorithm
    
    def f(d):
        f_val=obj_fun(observations,weights,d,wavelengths)
        return f_val
    
    opt_res = gp_minimize(f, [(0.0, 10.0)], n_calls=100)
    d_global=opt_res.x
    
    
    # iii) Accumulate residuals
    
    d_residuals_AR[k]=np.abs(d_AR-d_true)
    d_residuals_global[k]=np.abs(d_global[0]-d_true)


"""
    4. Compare results and ground truth ---------------------------------------
"""



# i) Assemble results

p_AR=np.sum(d_residuals_AR<10**(-3))/n_trials
p_global=np.sum(d_residuals_global<10**(-3))/n_trials

# ii) Print out results

print('The percentage of correctly estimated distances with the MILP approach is {} %'.format(100*p_AR))
print('The percentage of correctly estimated distances with the global solver is {} %'.format(100*p_global))


































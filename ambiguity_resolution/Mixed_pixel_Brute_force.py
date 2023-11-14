"""
The goal of this script is to illustrate the mixed pixel unmixing via an iterative 
approach. This serves as a test for examining the performance of the iterative
approach and for debugging the Mixed_pixel_resolution.py function.
For this, do the following:
    1. Definitions and imports
    2. Simulate data
    3. Solve the estimation problem once
    4. Second iteration to get the second distance
    5. Compare to function output
    6. Compare the results and ground truths
    7. Plots and illustrations

"""

"""
    1. Definitions and imports -----------------------------------------------
"""



# i) Imports


import sys

sys.path.insert(0, './AR_compilation')

import Support_funs_AR as sf
import numpy as np
import cvxpy as cp
import Ambiguity_resolution as AR


# ii) Basic definitions

n_obs=10

weights=np.array([1.6,1])
distances=np.array([1,2])
wavelengths=np.linspace(0.01,0.05,n_obs)

phase_variances=np.ones([n_obs])*0.0001




"""
    2. Simulate data ---------------------------------------------------------
"""

    
# i) Coefficient matrices

lambda_mat=np.diag(wavelengths)
lambda_mat_pinv=np.linalg.pinv(lambda_mat)
lambda_vec_pinv=np.diag(lambda_mat_pinv)

phase_std=np.sqrt(phase_variances)
phase_std_pinv=np.linalg.pinv(np.diag(phase_std))


# ii) Generate the data

observations, C_mat=sf.Generate_data_noisy(weights,distances,wavelengths,phase_variances)
z_obs=observations



"""
    3. Solve the estimation problem once -------------------------------------
"""


# i) Solve the problem

cons=['d_opt<=10']
optim_opts=sf.Setup_optim_options(n_obs, constraints=cons)

d,N,r=AR.Ambiguity_resolution(observations, wavelengths, phase_variances,optim_opts)


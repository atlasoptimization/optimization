"""
The goal of this script is to compare classical two color solution to an 
optimization based approach to distance estimation. 
For this, do the following:
    1. Definitions and imports
    2. Simulate date
    3. Two color closed form solution
    4. Optimization based solution
    5. Plots and illustrations
"""


"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from scipy.optimize import minimize
import sys
sys.path.append("/home/jemil/Desktop/Programming/Python/Optimization/MW_EDM_estimation/MW_Estimation_aux/")
import CiddorPy


# ii) Definitions

n_obs=5
wavelengths=np.linspace(0.55,1.05,n_obs)
sigma=1/wavelengths
P_true, T_true, U_true, C_true, D_true = 1013.25, 20, 100, 450, 50


"""
    2. Simulate date
"""

# i) Generate ground truth observations

n_vec=np.zeros([n_obs])
for k in range(n_obs):
    n_vec[k]=CiddorPy.CiddorHillGroup(sigma[k], P_true, T_true, U_true, C_true)


d_refr=D_true*n_vec


# ii) Add noise

w_noise=np.zeros([n_obs])
# w_noise=np.random.normal(0,50e-6,[n_obs])

d_obs=d_refr+w_noise



"""
    3. Two color closed form solution
"""

# i) Directly use closed form solution

A=(n_vec[0]-1)/(n_vec[1]-n_vec[0])
D_est_TC=d_obs[0]-A*(d_obs[1]-d_obs[0])
d_est_vec_TC=np.zeros([n_obs])
for k in range(n_obs):
    d_est_vec_TC[k]=D_true*CiddorPy.CiddorHillGroup(sigma[k], P_true, T_true, U_true, C_true)



"""
    4. Optimization based solution
"""

# i) Only allow temperature, pressure, Distance to be chosen

x_0=np.array([10,40])               #P,T,D
    
def loss_fun(x_vec):
    d_est_vec_opt=np.zeros([n_obs])
    for k in range(n_obs):
        d_est_vec_opt[k]=x_vec[1]*CiddorPy.CiddorHillGroup(sigma[k], P_true, x_vec[0], U_true, C_true)
        
    loss_val=np.linalg.norm(d_est_vec_opt-d_obs,1)
    return loss_val
    
x_guess = basinhopping(loss_fun, x_0, niter=1000, stepsize=1, disp=True)

# options={'gtol': 1e-08, 'eps': 1.4901161193847656e-09, 'maxiter': None, 'disp': True}
# x_guess = minimize(loss_fun,x_0,method='BFGS',options=options)

# x_guess = minimize(loss_fun,x_0,method='CG',options=options)

# options={'gtol': 1e-08, 'eps': 1.4901161193847656e-09, 'maxiter': None, 'xatol':10e-10, 'fatol':10e-10, 'disp': True}
# x_guess = minimize(loss_fun,x_0,method='Nelder-Mead',tol=10e-10)

x_opt=x_guess.x
d_est_vec_opt=np.zeros([n_obs])
for k in range(n_obs):
    d_est_vec_opt[k]=x_opt[1]*CiddorPy.CiddorHillGroup(sigma[k], P_true, x_opt[0], U_true, C_true)


"""
    5. Plots and illustrations
"""
print('True:', D_true)
print('TC:' , D_est_TC)
print('Opt:', x_guess.x)

print('Residuals TC', np.linalg.norm(d_obs-d_est_vec_TC,1))
print('Residuals Opt', np.linalg.norm(d_obs-d_est_vec_opt,1))




































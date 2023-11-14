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



"""
    4. Second iteration to get the second distance ---------------------------
"""


# i) General setup

n_disc_base= 21
n_disc=n_disc_base+2*n_obs

z_sol_temp=np.exp(4*np.pi*1j*d*lambda_vec_pinv)


# ii) Set up linearization
    
w_disc=np.logspace(-1,1,n_disc_base)
intersections=np.zeros(n_obs)

for k in range(n_obs):
    w_intersect_b= np.abs(np.imag(z_obs[k])/np.imag(z_sol_temp[k]))-0.01
    w_intersect_u= np.abs(np.imag(z_obs[k])/np.imag(z_sol_temp[k]))+0.01
    w_disc=np.append(w_disc,[w_intersect_b,w_intersect_u])
    
w_disc=np.sort(w_disc)

d_w=np.roll(w_disc,-1)-w_disc
d_w=np.delete(d_w,n_disc-1)

phi=np.zeros([n_obs,n_disc])

for k in range(n_obs):
    for l in range(n_disc):
        phi[k,l]=np.angle(z_obs[k] - w_disc[l]*z_sol_temp[k])

d_phi=np.roll(phi,-1,axis=1)-phi
d_phi=np.delete(d_phi,n_disc-1, 1)
phi_0=phi[:,0]


# iii) Create optimization variables

d_opt=cp.Variable(1,pos=True)
N_opt=cp.Variable(n_obs,integer=True)
w_opt=cp.Variable(n_disc-1,pos=True)
log_opt=cp.Variable(n_disc,boolean=True)

cons=['log_opt[0]==1']

for k in range(n_disc-1):
    cons=cons+['d_w[{}]*log_opt[{}]<=w_opt[{}]'.format(k,k+1,k)]
    cons=cons+['w_opt[{}]<=d_w[{}]*log_opt[{}]'.format(k,k,k)]

cons=cons+['d_opt<=10']

constraints_opt=[]
for cstr in cons:
    constraints_opt=constraints_opt+[eval(cstr)]


#objective=cp.Minimize(cp.norm(phase_std_pinv@(2*np.pi*(2*d_opt*lambda_vec_pinv-N_opt)-(phi_0+d_phi@w_opt)),p=1))
objective=cp.Minimize(cp.norm((2*np.pi*(2*d_opt*lambda_vec_pinv-N_opt)-(phi_0+d_phi@w_opt)),p=1))

# iv) Solve and extract

Optim_problem=cp.Problem(objective,constraints=constraints_opt)
Optim_solution=Optim_problem.solve(solver='GLPK_MI',max_iters=100, verbose=True)



"""
    5. Compare to function output --------------------------------------------
"""


# i) Objective function

def residuals(observations, weights,distances,wavelengths):
    
    observations_pred, _=sf.Generate_data(weights,distances,wavelengths)
    phi_pred=np.angle(observations_pred)
    phi_diff=np.angle(np.conj(observations)*observations_pred)
    
    return phi_diff, phi_pred

def obj_fun(observations, weights,distances,wavelengths):
    phi_residuals,_ = residuals(observations, weights,distances,wavelengths)
    norm_resid=np.linalg.norm(phi_residuals,1)
    return norm_resid



# ii) Alternative methods: Brute force

w_bf=w_disc
d_bf=np.zeros(n_disc)
obj_val_bf=np.zeros(n_disc)

observations_d1,_=sf.Generate_data([1],[d],wavelengths)

for k in range(n_disc):
    observations_residuals=observations-w_bf[k]*observations_d1
    d_bf[k],_,_=AR.Ambiguity_resolution(observations_residuals, wavelengths, phase_variances,optim_opts)
    obj_val_bf[k]=obj_fun(observations,[1,w_bf[k]],[d,d_bf[k]],wavelengths)



"""
    6. Compare the results and ground truths ---------------------------------
"""


# i) Objective function



# ii) Ground truth versus solution

residuals_gt,_=residuals(observations,weights,distances,wavelengths)
obj_val_gt=obj_fun(observations,weights,distances,wavelengths)


d_estimate=np.array([d,np.sum(d_opt.value)])
w_estimate=np.array([1,np.sum(w_opt.value)])

residuals_opti,_=residuals(observations,w_estimate,d_estimate,wavelengths)
obj_val_opti=obj_fun(observations,w_estimate,d_estimate,wavelengths)



"""
    7. Plots and illustrations -----------------------------------------------
"""


# i) Print out solutions

print(distances)
print(d_estimate)

print(weights)
print(w_estimate)

print(obj_val_gt)
print(obj_val_opti)
































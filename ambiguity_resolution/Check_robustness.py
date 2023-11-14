"""
The goal of this script is to check if the L1 norm penalty favours a robust,
dominant solution or a mixture of distances.
For this, do the following:
    1. Definitions and imports
    2. Simulate observations
    3. Evaluate objective function values and bounds
    4. Plots and illustrations  
"""

"""
    1. Definitions and imports
"""

import numpy as np
import matplotlib.pyplot as plt


w_2,w_1=np.sort(np.random.uniform(0,2,2))
d_2,d_1=np.sort(np.random.uniform(0,10,2))

n_obs=30
wavelengths=np.logspace(-2,-4,n_obs)


"""
    2. Simulate observations
"""

phi_1=np.zeros(n_obs)
phi_2=np.zeros(n_obs)

def calculate_phi_hat(d):
    phi_hat=np.zeros([n_obs])
    for k in range(n_obs):
        phi_hat[k]=np.arctan2(np.sin(2*np.pi*(d/wavelengths[k])),np.cos(2*np.pi*(d/wavelengths[k])))
        
    return phi_hat
    
phi_1=calculate_phi_hat(d_1)
phi_2=calculate_phi_hat(d_2)


phi_obs=np.zeros([n_obs])
sinsum=np.zeros([n_obs])
cossum=np.zeros([n_obs])

for k in range(n_obs):
    sinsum[k]=w_1*np.sin(phi_1[k])+w_2*np.sin(phi_2[k])
    cossum[k]=w_1*np.cos(phi_1[k])+w_2*np.cos(phi_2[k])
    phi_obs[k]=np.arctan2(sinsum[k],cossum[k])


"""
    3. Evaluate objective function values and bounds
"""

def dist_fun_l1(phi1,phi2):
    min_dist=min(np.abs(phi1-phi2),np.abs(phi1-phi2-2*np.pi),np.abs(phi1-phi2+2*np.pi))
    
    return min_dist

def obj_fun(phi_obs, phi_hat):
    n=np.shape(phi_obs)[0]
    
    dist_sum=0
    for k in range(n):
        dist_sum=dist_sum+dist_fun_l1(phi_obs[k],phi_hat[k])
        
    return dist_sum
        


n_discrete=10000
d_test=np.linspace(0,10,n_discrete)

obj_fun_vals=np.zeros([n_discrete])
for k in range(n_discrete):
    obj_fun_vals[k]=obj_fun(phi_obs,calculate_phi_hat(d_test[k]))

obj_fun_d_1=obj_fun(phi_obs,calculate_phi_hat(d_1))
obj_fun_d_2=obj_fun(phi_obs,calculate_phi_hat(d_2))

min_obj_fun_bf=np.min(obj_fun_vals)

print(w_1/w_2)
print(obj_fun_d_1,obj_fun_d_2,np.mean(obj_fun_vals), min_obj_fun_bf)

Delta_psi=np.arccos(-w_2/w_1)-np.pi/2
print(n_obs*Delta_psi)

"""
    4. Plots and illustrations  
"""
plt.figure(1)
plt.scatter(phi_1,np.zeros([n_obs]))

plt.figure(2)
plt.scatter(d_test,obj_fun_vals)





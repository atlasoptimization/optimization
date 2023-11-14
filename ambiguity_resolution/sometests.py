"""
The goal of this script is to try out some optimality criterions and see if they
are suitable for defining solutions to the mixed pixel problem.

First thing we will investigate is if the individual phases associated to two
simultaneously measured locations can somehow be extracted from the nonlinear 
min problem |Aw-z_ob|->min in complex phases A and weights w.

For this, do the following
    1. Definitions and imports
    2. Simulate some data
    3. Check optimality
    4. Add noise, check again
    5. Plots and illustrations


"""


"""
    1. Definitions and imports
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt

n_wave=10
n_mix=2

d=np.linspace(10,11,n_mix)
d=(np.random.normal(3,1,[n_mix]))**2
lambda_vec=np.logspace(10**(-4),10**(-1),n_wave)


w=np.linspace(1,2,n_mix)

"""
    2. Simulate some data
"""


# For each of the wavelengths, calculate the mixture complex number
A=np.zeros([n_wave,n_mix],dtype='complex')
Phi_mat=np.zeros([n_wave,n_mix],dtype='complex')


for k in range(n_wave):
    for l in range(n_mix):
        Phi_mat[k,l]=2*np.pi*(d[l]/lambda_vec[k])
        A[k,l]=np.exp(1j*Phi_mat[k,l])
        
z_obs=A@w


"""
    3. Check optimality
"""

def obj_fun(z_obs, A, w):
    obj_val=(np.linalg.norm(z_obs-A@w))**2
    
    return obj_val

def obj_fun_real(z_obs_re,z_obs_im, A_re, A_im,w):
    obj_val=(np.linalg.norm(z_obs_re-A_re@w))**2+(np.linalg.norm(z_obs_im-A_im@w))**2
    
    return obj_val

def obj_fun_scopt(x):
    w=x[:n_mix]
    A=np.reshape(x[n_mix:],[2*n_wave,n_mix])
    
    z_obs_tilde=np.hstack((np.real(z_obs),np.imag(z_obs)))
    
    obj_val=(np.linalg.norm(z_obs_tilde-A@w))**2
    
    return obj_val

print(obj_fun(z_obs,A,w))

grad_w=2*(z_obs-A@w).T@A
grad_phi=np.multiply((2*1j*A),np.outer((z_obs-A@w),w.T))


# Try solve self by blockwise descent

w_0=np.ones([n_mix,1])
Phi_0=np.mod(np.random.normal(0,1,[n_wave,n_mix]),2*np.pi)-np.ones([n_wave,n_mix])*np.pi
A_0=np.exp(1j*Phi_0)

Phi_mat_guess=Phi_0
A_guess=np.copy(A_0)

A_guess_tilde=np.vstack((np.real(A_guess),np.imag(A_guess)))
z_obs_tilde=np.hstack((np.real(z_obs),np.imag(z_obs)))

n_iter=100
for k in range(n_iter):
    
    # First estimate w with nnls
    w_guess=scopt.nnls(A_guess_tilde,z_obs_tilde,maxiter=10)[0]
    
    # Then estimate The optimal phis
    grad_phi_guess=np.multiply((2*1j*A_guess),np.outer((z_obs-A_guess@w_guess),w_guess.T))
    Phi_mat_guess=Phi_mat_guess-0.001*grad_phi_guess
    A_guess=np.exp(1j*Phi_mat_guess)
    
print(obj_fun(z_obs,A_guess,w_guess))

x0=np.hstack((np.ravel(w_0),np.ravel(np.real(A_0)), np.ravel(np.imag(A_0))))
x_scopt=scopt.minimize(obj_fun_scopt,x0)

print(obj_fun_scopt(x_scopt.x))



# objective function incorporating unit norm, ie a[k,l]=e^iphi
def obj_fun_scopt_constrained(x):
    w=x[:n_mix]
    phi=x[n_mix:]
    
    Phi_mat=np.reshape(phi,[n_wave,n_mix])
    
    A=np.exp(1j*Phi_mat)
    
    A_tilde=np.vstack((np.real(A),np.imag(A)))
    
    z_obs_tilde=np.hstack((np.real(z_obs),np.imag(z_obs)))
    obj_val=(np.linalg.norm(z_obs_tilde-A_tilde@w))**2
    
    return obj_val

x0_constrained=np.hstack((np.ravel(w_0),np.ravel(Phi_0)))


bnds=(())
for k in range(n_mix):
    bnds=bnds+((0,None),)
    
for k in range(n_mix*n_wave):
    bnds=bnds+((None,None),)


x_scopt_constrained=scopt.minimize(obj_fun_scopt_constrained,x0_constrained, bounds=bnds)
print(obj_fun_scopt_constrained(x_scopt_constrained.x))                                   
                                   
w_guess_scopt=(x_scopt_constrained.x)[:n_mix]
Phi_guess_scopt=np.reshape((x_scopt_constrained.x)[n_mix:],[n_wave,n_mix])
A_guess_scopt=np.exp(1j*Phi_guess_scopt)

print('Discrepancies in w, A)')
print(np.linalg.norm(w_guess_scopt-w))
print(np.linalg.norm(A_guess_scopt-A))



"""
    4. Add noise, check again
"""



"""
    5. Plots and illustrations
"""


plt.scatter(np.real(z_obs),np.imag(z_obs))





















































































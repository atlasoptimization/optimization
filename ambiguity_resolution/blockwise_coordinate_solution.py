"""
The goal of this script is to try out some algortihms for calculating estimations
of solutions to the mixed pixel problem.

Throughout we will try to via different approximations jointly solve a mixture of
an L2 norm minimization and L1 norm minimizations. They represent the suitability 
of the decomposition into mixed pixels and the plausibility of the distance estimations
respectively.

For this, do the following
    1. Definitions and imports
    2. Simulate some data
    3. Set up optimization
    4. Blockwise coordinate descent
    5. Plots and illustrations


"""


"""
    1. Definitions and imports -----------------------------------------------
"""


import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import scipy.optimize as scopt

n_wave=10
n_mix=2

d=np.linspace(10,11,n_mix)
d=(np.random.normal(3,1,[n_mix]))**2
lambda_vec=np.logspace(-1,0,n_wave)

N_true=np.zeros([n_wave,n_mix])
for k in range(n_wave):
    for l in range(n_mix):
        N_true[k,l]=np.floor_divide(d[l],lambda_vec[k])

w=np.linspace(1,2,n_mix)


"""
    2. Simulate some data ----------------------------------------------------
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
    3. Set up optimization ---------------------------------------------------
"""


# i) Intital solutions and real decomposition

w_0=np.ones([n_mix,1])
Phi_0=np.mod(np.random.normal(0,1,[n_wave,n_mix]),2*np.pi)-np.ones([n_wave,n_mix])*np.pi
A_0=np.exp(1j*Phi_0)

Phi_mat_guess=Phi_0
A_guess=np.copy(A_0)

A_guess_tilde=np.vstack((np.real(A_guess),np.imag(A_guess)))
z_obs_tilde=np.hstack((np.real(z_obs),np.imag(z_obs)))


# ii) Objective function incorporating unit norm, ie a[k,l]=e^iphi
def obj_fun(x):
    w=x[:n_mix]
    phi=x[n_mix:]
    
    Phi_mat=np.reshape(phi,[n_wave,n_mix])
    
    A=np.exp(1j*Phi_mat)
    
    A_tilde=np.vstack((np.real(A),np.imag(A)))
    
    z_obs_tilde=np.hstack((np.real(z_obs),np.imag(z_obs)))
    obj_val=(np.linalg.norm(z_obs_tilde-A_tilde@w))**2
    
    return obj_val

x0=np.hstack((np.ravel(w_0),np.ravel(Phi_0)))


bnds=(())
for k in range(n_mix):
    bnds=bnds+((0,None),)
    
for k in range(n_mix*n_wave):
    bnds=bnds+((None,None),)




"""
    4. Blockwise coordinte descent -------------------------------------------
"""


# i) Get soluton for w, phi
opts={"maxiter":300,"disp":True}
opt_result=scopt.minimize(obj_fun,x0, bounds=bnds,options=opts)
x_scopt=opt_result.x
print(obj_fun(x_scopt))                                   
                                   
w_guess=(x_scopt)[:n_mix]
Phi_guess=np.reshape((x_scopt)[n_mix:],[n_wave,n_mix])
A_guess=np.exp(1j*Phi_guess)

print('Discrepancies in w, A)')
print(np.linalg.norm(w_guess-w))
print(np.linalg.norm(A_guess-A))


# ii) Estimate d, N
N_opt=cp.Variable(n_wave*n_mix,integer=True)
d_opt=cp.Variable(n_mix, nonneg=True)

Lambda_mat=np.diag(lambda_vec)
Lambda_mat_pinv=np.linalg.pinv(np.diag(lambda_vec))
Lambda_vec_pinv=np.diag(Lambda_mat_pinv)

# N_hat_vec=np.zeros([0])
# for k in range(n_mix):
#     N_hat_vec=np.hstack((N_hat_vec,d_opt[k]*Lambda_vec_pinv))

if n_mix==1:
        objective=cp.Minimize(cp.norm(2*np.pi*(d_opt*Lambda_vec_pinv-N_opt)-np.ravel(Phi_guess),p=1))
elif n_mix==2:
        objective=cp.Minimize(cp.norm(2*np.pi*(cp.hstack((d_opt[0]*Lambda_vec_pinv,d_opt[1]*Lambda_vec_pinv))-N_opt)-np.ravel(Phi_mat),p=1))
else:
    print('Inference only for n_mix <=2')
            
            
# objective=cp.Minimize(cp.norm(2*np.pi*(N_hat_vec-N_opt)-np.ravel(Phi_guess),p=1))


cons=[]
for k in range(n_wave*n_mix):
    cons=cons+[N_opt[k]>=0]
    
prob=cp.Problem(objective,constraints=cons)
obj_val_cvxpy=prob.solve(solver='GLPK_MI',max_iters=2, verbose=True)




# B=np.random.normal(0,1,size=[n_wave,n_wave])

# x_opt=cp.Variable(shape=n_wave-1, pos=True)
# q_opt=cp.Variable(shape=1,integer=True)

# objective=cp.Minimize(cp.norm(B@cp.hstack((q_opt,x_opt))-B[:,0]**2,p=1))
# prob=cp.Problem(objective,[q_opt>=-5,cp.sum(x_opt)+q_opt==10])

# x_cvxpy_guess=prob.solve()


"""
    5. Plots and illustrations
"""



plt.scatter(np.real(z_obs),np.imag(z_obs))














































































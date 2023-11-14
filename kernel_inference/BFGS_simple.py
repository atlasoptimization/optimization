# The goal of this script is to test the BFGS approach to unconstrained kernel 
# inference. For this, do the following:
# 1. Definitions and imports
# 2. Simulation of observations
# 3. Preparation for optimization
# 4. BFGS iteration
# 5. Plots and illustrations


# 1. Definitions and imports

# i) Imports
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

# ii) Simulation
n_time=100
time=np.linspace(0,1,n_time)

n_simu=2
d=0.2
d_guess=0.5

# iii) Observations
n_obs=10
ind_obs=range(0,n_time-1,n_obs)
t_obs=time[ind_obs]

# iv) Inference
n_exp=5
n_iter=100

# 2. Simulation of observations

# i) Correlation structure and mean
def cov_fun(s,t):
    cov_val=np.exp(-((s-t)/d)**2)
    return cov_val

C_full=np.zeros([n_time,n_time])
for i in range(n_time):
    for j in range(n_time):
        C_full[i,j]=cov_fun(time[i],time[j])
      
mu=np.zeros([n_time])

# ii) Simulation of observations      
x_simu=np.zeros([n_time,n_simu])        
for k in range(n_simu):
    x_simu[:,k]=np.random.multivariate_normal(mu,C_full)
    
x_obs=x_simu[ind_obs,:]


# 3. Preparation for optimization

# i) Prior and decomposition
def cov_fun_guess(s,t):
    cov_val=np.exp(-((s-t)/d_guess)**2)
    return cov_val

C_guess=np.zeros([n_time,n_time])
for i in range(n_time):
    for j in range(n_time):
        C_guess[i,j]=cov_fun_guess(time[i],time[j])
        
[Phi,Lambda,V]=np.linalg.svd(C_guess,hermitian=True)
Lambda=np.diag(Lambda)

Psi=Phi[ind_obs,:]
Psi=Psi[:,range(0,n_exp)]


# ii) Gradient function
def pg(gamma):
    return Psi@gamma@Psi.T
def pg_inv(gamma): 
    return np.linalg.pinv(Psi@gamma@Psi.T)

S_emp= (1/n_simu)*x_obs@x_obs.T
S= (1/n_simu)*x_simu@x_simu.T
gamma_prior=Lambda[0:n_exp,0:n_exp]

def grad(gamma, S_emp):
    gamma=np.reshape(gamma,[n_exp,n_exp])
    
    pg_temp=pg(gamma)
    pg_inv_temp=pg_inv(gamma)
    
    g1=-n_simu*Psi.T@pg_inv_temp@Psi
    g2=Psi.T@(pg_inv_temp@S_emp@pg_inv_temp)@Psi
    g3=-Psi.T@(np.eye(n_obs)-pg_temp@pg_inv_temp)@S_emp@(pg_inv_temp@pg_inv_temp)@Psi
    g4=-Psi.T@((pg_inv_temp@pg_inv_temp)@S_emp@(np.eye(n_obs)-pg_inv_temp@pg_temp))@Psi
    g_data=g1+g2+g3+g4
    g_data=(1/2)*(g_data+g_data.T)
    
    g5=np.linalg.pinv(gamma)
    g6=np.linalg.pinv(gamma_prior)
    g_prior=g5+g6
    
    return g_data, g_prior

def grad_mod(gamma):
    g_data,g_prior=grad(gamma,S_emp)
    gradient=np.reshape(g_data+g_prior,[n_exp**2])
    return gradient

    
# iii) Initializations
gamma_0=gamma_prior
B_0_inv=np.kron(gamma_prior,gamma_prior)


# 4. BFGS iteration

def pseudodet(Symmat):
    [U,S,V]=np.linalg.svd(Symmat)
    pseudodet=np.prod(S[S>10**(-10)])
    return pseudodet

def obj_fun(gamma):
    gamma=np.reshape(gamma,[n_exp,n_exp])
    C_gamma=Psi@gamma@Psi.T
    obj_val=n_simu*np.log(pseudodet(C_gamma))+np.trace(S_emp@(np.linalg.pinv(C_gamma)))-np.log(pseudodet(gamma))+np.trace(np.linalg.pinv(gamma_prior)@gamma)
    return obj_val


gamma=gamma_0
B_inv=B_0_inv

for k in range(n_iter):
    gamma=np.reshape(gamma,[n_exp**2])
    g_data,g_prior=grad(gamma,S_emp)
    gradient=np.reshape(g_data+g_prior,[n_exp**2])
    p=-B_inv@gradient
    
    alpha=opt.line_search(obj_fun,grad_mod,gamma,p)[0]
    s=alpha*p
    gamma=gamma+s
    
    g_data,g_prior=grad(gamma,S_emp)
    gradient_new=np.reshape(g_data+g_prior,[n_exp**2])
    y=gradient_new-gradient
    
    B_inv=B_inv+(np.dot(s,y)+np.dot(y,B_inv@y))/(np.dot(s,y)**2)*np.outer(s,s) - (1/(np.dot(s,y)))*(np.outer(B_inv@y,s)+np.outer(s,y.T@B_inv.T))





# 5. Plots and illustrations

S_psi=np.linalg.pinv(Psi)@(S_emp)@np.linalg.pinv(Psi.T)
# Original KI for comparison
gamma=gamma_prior
r=1
for k in range(n_iter):
    gamma= gamma+(1/(1+r))*((r-1)*gamma+S_psi-r*gamma@np.linalg.pinv(gamma_prior)@gamma)




plt.plot(time,x_simu)








































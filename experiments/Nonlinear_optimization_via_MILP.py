"""
The goal of this script is to test the capabilities of mixed integer linear programming
for solving nonlinear optimization problems. For this, do the following:
    1. Definitions and imports
    2. Set upt optimization problem
    3. Solve and illustrate solution    
    
The function to be optimized is x_1^2+x_2 with some linear constraints on it.

min x_1^2+x_2
s.t. x_2>=2
     x_1+x_2=3
"""



"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import matplotlib as plt
import cvxpy as cp


# ii) Basic quantities

n=25
t=np.linspace(0,4,n)
d_t=np.roll(t,-1)-t
d_t=np.delete(d_t,n-1)

def objective_true(x_1,x_2):
    return x_1**2+x_2



"""
    2. Set upt optimization problem
"""


# i) Discretization

f=t**2
d_f=np.roll(f,-1)-f
d_f=np.delete(d_f,n-1)

slope=d_f/d_t


# ii) Linear inequalities

x_1_opt=cp.Variable(n-1,pos=True)
x_2_opt=cp.Variable(1)
w_opt=cp.Variable(n,boolean=True)

cons=['w_opt[0]==1']

for k in range(n-1):
    cons=cons+['d_t[{}]*w_opt[{}]<=x_1_opt[{}]'.format(k,k+1,k)]
    cons=cons+['x_1_opt[{}]<=d_t[{}]*w_opt[{}]'.format(k,k,k)]

cons=cons+['x_2_opt>=2']
cons=cons+['cp.sum(x_1_opt)+x_2_opt==3']

constraints_opt=[]
for cstr in cons:
    constraints_opt=constraints_opt+[eval(cstr)]


# iii) Objective function

objective=cp.Minimize(slope.T@x_1_opt+x_2_opt)




"""
    3. Solve and illustrate solution
"""


Optim_problem=cp.Problem(objective,constraints=constraints_opt)
Optim_solution=Optim_problem.solve(solver='GLPK_MI',max_iters=300, verbose=True)

print(x_1_opt.value)
print(x_2_opt.value)
print(np.sum(x_1_opt.value))
print(Optim_solution)

x_1_guess=np.sum(x_1_opt.value)
x_2_guess=x_2_opt.value

print(objective_true(x_1_guess,x_2_guess))




















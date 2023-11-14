"""
The goal of this script is to solve a small SDP. This code is adjustable to come 
up with an initial guess of a posidive semidefinite matrix satisfying linear
constraints.

The script is meant solely for educational and illustrative purposes. Written by
Jemil Avers Butt, Atlas optimization GmbH, www.atlasoptimization.com.

"""



"""
    1. Imports and definitions
"""


# i) Imports

import numpy as np
import cvxpy as cp
import copy as copy
import matplotlib.pyplot as plt


# ii) Definitions

n_dim=5


"""
    2. Formulate the semidefinite problem
"""


# i) Define variables

opt_var=cp.Variable((n_dim,n_dim),PSD=True)


# ii) Define constraints

# cons=[opt_var[1,1]==1]
cons=[np.ones([1,n_dim**2])@cp.vec(opt_var)==1]


# iii) Define objective function

objective=cp.Minimize(cp.trace(opt_var))



"""
    3. Assemble the solution
"""


# i) Solve problem

prob=cp.Problem(objective,cons)
prob.solve(verbose=True)


# ii) Extract solution

x_opt=opt_var.value










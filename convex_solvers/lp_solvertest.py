"""
The goal of this script is to test atlas_optimization.lp_solve on a range of
different randomly generated lp's and compare the results with output from cvxpy.
For this, do the following:
    1. Definitions and imports
    2. Generate lp's randomly
    3. Solve via cvxpy
    4. Solve via atlas_optimization.lp_solve
    5. Plots and illustrations
"""

"""
    1. Definitions and imports
"""


# i) Imports

import numpy as np
import cvxpy as cp
import atlas_optimization
import matplotlib.pyplot as plt
import timeit
import copy



# ii) Definitions

n=100
n_simu=1
n_eq=10
n_ineq=10            # + n nonnegativity constraints



"""
    2. Generate lp's randomly
"""


# i) Random vectors

c=np.random.normal(0,1,[n,1])**2
b=np.random.normal(0,1,[n_eq,1])
h=np.vstack((np.zeros([n,1]),np.random.normal(0,1,[n_ineq,1])))


# ii) Random matrices

A=np.random.normal(0,1,[n_eq,n])
G=np.block([[np.eye(n)],[np.random.normal(0,1,[n_ineq,n])]])



"""
    3. Solve via cvxpy
"""


# i) Problem definition

x_var=cp.Variable([n,1])
cons=[A@x_var==b,G@x_var>=h]

objective=cp.Minimize(c.T@x_var)
lp_problem=cp.Problem(objective,cons)


# ii) Solve problem

lp_problem.solve(verbose=True)
print(lp_problem.value)
# print(x_var.value)


"""
    4. Solve via atlas_optimization.lp_solve
"""

start_time=timeit.default_timer()
x_opt, opt_logfile=atlas_optimization.lp_solve(c,A,b,G,h)
stop_time=timeit.default_timer()

print('Runtime is', stop_time-start_time)
plt.plot(x_var.value-x_opt)



# """
#     5. Plots and illustrations
# """







# # Primitive tests
# x=np.ones([2,1])
# l=np.ones([2,1])
# nu=np.ones([1,1])

# A=np.ones([1,2])
# b=np.ones([1,1])
# c=np.array([[1],[5]])
# G=np.eye(2)
# h=np.zeros([2,1])

# n_s=2
# m_1_s=1


# # x_opt, opt_logfile=atlas_optimization.lp_solve_standard(c,A,b)
# x_opt, opt_logfile=atlas_optimization.lp_solve(c,A,b,G,h)






# # i) Random vectors

# c=np.random.normal(0,1,[n,1])**2
# b=np.random.normal(0,1,[n_eq,1])


# # ii) Random matrices

# A=np.random.normal(0,1,[n_eq,n])



# """
#     3. Solve via cvxpy
# """


# # i) Problem definition

# x_var=cp.Variable([n,1],nonneg=True)
# cons=[A@x_var==b]

# objective=cp.Minimize(c.T@x_var)
# lp_problem=cp.Problem(objective,cons)


# # ii) Solve problem

# lp_problem.solve(verbose=True,solver=cp.ECOS)
# print(lp_problem.value)


# """
#     4. Solve via atlas_optimization.lp_solve
# """

# start_time=timeit.default_timer()
# x_opt,opt_logfile=atlas_optimization.lp_solve_standard(c,A,b)
# stop_time=timeit.default_timer()

# print('Runtime is', stop_time-start_time)
# plt.plot(x_var.value-x_opt)





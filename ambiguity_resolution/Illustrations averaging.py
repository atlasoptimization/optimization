"""
The goal of this script is to illustrate how the averaging of phasaes works in 
a complexvalued setting according to the IQ model. This is to be compared to
a linear averaging procedure.
For this, do the following:
    1. Definitions and imports
    2. Simulate data in complex space
    3. Compare to other averaging procedures
    4. Plots and illustrations
"""


"""
    1. Definitions and imports
"""


import numpy as np
import matplotlib.pyplot as plt


n=100


"""
    2. Simulate data in complex space
"""

t=np.linspace(0,1,n)
phi_1=np.mod(np.linspace(0,4*np.pi,n),2*np.pi)-np.pi*np.ones([n])
phi_2=np.mod(np.linspace(0,4*np.pi,n),2*np.pi)-np.pi*np.ones([n])

w_1=2
w_2=1

Supos_mat=np.zeros([n,n])
sinsum=np.zeros([n,n])
cossum=np.zeros([n,n])

for k in range(n):
    for l in range(n):
        sinsum[k,l]=w_1*np.sin(phi_1[k])+w_2*np.sin(phi_2[l])
        cossum[k,l]=w_1*np.cos(phi_1[k])+w_2*np.cos(phi_2[l])
        Supos_mat[k,l]=np.arctan2(sinsum[k,l],cossum[k,l])






"""
    3. Compare to other averaging procedures
"""

Linsum_mat=np.zeros([n,n])
q_1=w_1/(w_1+w_2)
q_2=w_2/(w_1+w_2)


for k in range(n):
    for l in range(n):
        Linsum_mat[k,l]=q_1*phi_1[k]+q_2*phi_2[l]




"""
    4. Plots and illustrations
"""


middle_index=np.round(n/2)

w,h=plt.figaspect(0.3)
fig1 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig1.add_gridspec(2, 3)


f1_ax1 = fig1.add_subplot(gs1[0,0])
f1_ax1.plot(t,Supos_mat[0,:])

f1_ax2 = fig1.add_subplot(gs1[0,1])
f1_ax2.plot(t,Supos_mat[40,:])


f1_ax3 = fig1.add_subplot(gs1[0,2])
f1_ax3.plot(t,Supos_mat[-20,:])



f1_ax4 = fig1.add_subplot(gs1[1,0])
f1_ax4.plot(t,Linsum_mat[0,:])

f1_ax5= fig1.add_subplot(gs1[1,1])
f1_ax5.plot(t,Linsum_mat[40,:])


f1_ax6 = fig1.add_subplot(gs1[1,2])
f1_ax6.plot(t,Linsum_mat[-20,:])











w,h=plt.figaspect(0.3)
fig2 = plt.figure(dpi=400,constrained_layout=True,figsize=(w,h))
gs1 = fig2.add_gridspec(2, 3)


f2_ax1 = fig2.add_subplot(gs1[0,0])
f2_ax1.plot(t,Supos_mat[:,0])

f2_ax2 = fig2.add_subplot(gs1[0,1])
f2_ax2.plot(t,Supos_mat[:,40])


f2_ax3 = fig2.add_subplot(gs1[0,2])
f2_ax3.plot(t,Supos_mat[:,-20])


f2_ax4 = fig2.add_subplot(gs1[1,0])
f2_ax4.plot(t,Linsum_mat[:,0])

f2_ax5= fig2.add_subplot(gs1[1,1])
f2_ax5.plot(t,Linsum_mat[:,40])


f2_ax6 = fig2.add_subplot(gs1[1,2])
f2_ax6.plot(t,Linsum_mat[:,-20])














































import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import CiddorPy
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, rosen

FONT_SIZE = 14
FIG_SIZE = (6, 6)
rc('font', size=FONT_SIZE)
rc('axes', titlesize=FONT_SIZE)


K_SAMPLES = 10
WAVELENGTH = np.linspace(550, 1050, K_SAMPLES, endpoint=True)
NU = 1 / (1e-3 * WAVELENGTH)

def model(X):
    n = CiddorPy.CiddorHillGroup(NU, X[0], X[1], X[2], X[3])
    result = n * X[4]
    return result

def objective(X):
    res = D_OBS - model(X)
    obj = np.sum(res**2)
    return obj


P_TRUE, T_TRUE, H_TRUE, C_TRUE = 1013.25, 20, 50, 450
D_TRUE = .50
# X = np.array([P_TRUE, T_TRUE, H_TRUE, C_TRUE, D_TRUE])
# X = X.reshape(len(X), 1)

X = [P_TRUE, T_TRUE, H_TRUE, C_TRUE, D_TRUE]

D_OBS = model(X)

x0 = [1012.25, 19, 48, 430, 0.50]
# x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
bnds = [(1011.25, 1014.25), (19, 21), (49, 51), (445, 455), (.49, .51)]
res = minimize(objective, (1012.25, 19, 48, 430, 0.50), method='SLSQP', bounds=bnds, tol=1e-9)
print(res.x)

val = D_OBS - model(res.x)
plt.plot(val, 'o')
plt.show()
# res = minimize(model, x0, method='Nelder-Mead', tol=1e-6)


# sys.exit()
#
#
# P_TRUE, T_TRUE, H_TRUE, C_TRUE = 1013.25, 20, 50, 450
# METEO_TRUE = [P_TRUE, T_TRUE, H_TRUE, C_TRUE]
# D_TRUE = 50
#
# D_OBS = model(NU, P_TRUE, T_TRUE, H_TRUE, C_TRUE, D_TRUE)
# D_OBS = D_OBS.reshape(K_SAMPLES, 1)
#
# P_0, T_0, H_0, C_0 = 1012.25, 18, 45, 430
# D_0 = 48
# X = np.array([P_0, T_0, H_0, C_0, D_0])
# X = X.reshape(len(X), 1)
# A = np.zeros((K_SAMPLES, len(X)))
#
# eps = 1e-5
# max_epochs = np.int(1e5)
#
# for epoch in range(max_epochs):
#
#     print(epoch)
#
#     x_p, x_t, x_h, x_c, x_d = X[0], X[1], X[2], X[3], X[4]
#
#     dxp_plus = model(NU, x_p + eps, x_t, x_h, x_c, x_d)
#     dxp_minus = model(NU, x_p - eps, x_t, x_h, x_c, x_d)
#     dfdp = (dxp_plus - dxp_minus) / (2*eps)
#
#     dxt_plus = model(NU, x_p, x_t + eps, x_h, x_c, x_d)
#     dxt_minus = model(NU, x_p, x_t - eps, x_h, x_c, x_d)
#     dfdt = (dxt_plus - dxt_minus) / (2*eps)
#
#     dxh_plus = model(NU, x_p, x_t, x_h + eps, x_c, x_d)
#     dxh_minus = model(NU, x_p, x_t, x_h + eps, x_c, x_d)
#     dfdh = (dxh_plus - dxh_minus) / (2*eps)
#
#     dxc_plus = model(NU, x_p, x_t, x_h, x_c + eps, x_d)
#     dxc_minus = model(NU, x_p, x_t, x_h, x_c - eps, x_d)
#     dfdc = (dxc_plus - dxc_minus) / (2*eps)
#
#     dxd_plus = model(NU, x_p, x_t, x_h, x_c, x_d + eps)
#     dxd_minus = model(NU, x_p, x_t, x_h, x_c, x_d - eps)
#     dfdd = (dxd_plus - dxd_minus) / (2*eps)
#
#     A[:, 0] = np.array(dfdp)
#     A[:, 1] = np.array(dfdt)
#     A[:, 2] = np.array(dfdh)
#     A[:, 3] = np.array(dfdc)
#     A[:, 4] = np.array(dfdd)
#
#     print(A)
#
#     D_EST = model(NU, x_p, x_t, x_h, x_c, x_d)
#     D_EST = D_EST.reshape(K_SAMPLES, 1)
#     dl = D_OBS - D_EST
#
#     # dx = np.dot(np.linalg.pinv(A), dl)  #replace with (ATA)-1AT
#     plt.imshow(A)
#
#     dx = np.linalg.inv(A.T @ A) @ A.T @ dl
#
#     X = X + dx
#
#     if np.max(np.abs(dx)) < 1e-10:
#         print("Converged at epoch: ", epoch)
#         break
#
#     print(X)
#
# print(X)
#
# # cov_matrix = np.linalg.inv(np.dot(A.transpose(), A))
#
# d_new = model(NU, 1019.54, 21.81, 45, 428.76, 50)
#
# print(mean_squared_error(D_OBS, d_new))
#

import sys
import CiddorPy
import numpy as np
from matplotlib import rc
from scipy.stats.distributions import chi2
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()

FONT_SIZE = 12
FIG_SIZE = (8, 4)
rc('font', size=FONT_SIZE)
rc('axes', titlesize=FONT_SIZE)
params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)


# MODEL
def model(pressure, temperature, humidity, carbondioxide, distance):
    nu = 1 / (1e-3 * wavelength)
    n = CiddorPy.CiddorHillGroup(nu, pressure, temperature, humidity, carbondioxide)
    opticalpathlength = n * distance
    return opticalpathlength.reshape(-1, 1)


# OBSERVATION EQUATION CONDITIONS
n_lambda = 2  # number of observations at different wavelengths
n_unknowns = 5  # number of unknowns i.e. P, T, U, C, D
n_dim = n_lambda + n_unknowns  # (n, 1) dimensions for the vectors later
wavelength = np.array([550, 1050] )# np.linspace(550, 1050, n_lambda, endpoint=True)

# GENERATING OBSERVATION VALUES
P_true, T_true, U_true, C_true, D_true = 1013.25, 20, 0, 450, 50
X_true = np.array([P_true, T_true, U_true, C_true, D_true])  # used later for chi2 test
# returns optical path length, wavelength values fed inside def_model
D_obs = model(P_true, T_true, U_true, C_true, D_true)
# std_D_obs = 50e-6 * np.ones(n_lambda)
std_D_obs = 50e-15 * np.ones(n_lambda)
noise = std_D_obs * rng.standard_normal(n_lambda)
noise = noise.reshape(-1, 1)
D_obs = np.add(D_obs, noise)

# PRIOR VALUES
P_prior, std_P_prior = 1005.25, 2 # in hPa
T_prior, std_T_prior = 15, 1e-9  # in degree Celsius
U_prior, std_U_prior = 0, 1e-12  # relative humidity in percentage
C_prior, std_C_prior = 450, 1e-9  # in ppm
D_prior, std_D_prior = 500, 4  # in meters

# INITIAL APPROXIMATION AND PRIOR VECTOR
X_prior = np.array([P_prior, T_prior, U_prior, C_prior, D_prior]).reshape(-1, 1)
std_X_prior = np.array([std_P_prior, std_T_prior, std_U_prior, std_C_prior, std_D_prior]).reshape(-1, 1)
X = np.copy(X_prior)  # initial approximation same as the prior values

# A-MATRIX INITIALIZATION
A = np.zeros((n_lambda, n_unknowns))
A = np.concatenate((A, np.identity(n_unknowns)))

# WEIGHT MATRIX INITIALIZATIONS
var_D_obs = np.square(std_D_obs).reshape(-1, 1)
var_X_prior = np.square(std_X_prior).reshape(-1, 1)
var_diag_elements = np.concatenate((var_D_obs, var_X_prior)).reshape(-1)
SIGMA_LL = np.diag(var_diag_elements)
variance_factor = 1  # unit variance (or) np.median(np.diag(SIGMA_LL))
Q_LL = SIGMA_LL / variance_factor
P = np.linalg.inv(Q_LL)

# COVARIANCE AND CORRELATION MATRIX INITIALIZATIONS
variance_factor_est = 0  # used in global quality test
SIGMA_XX = np.zeros((n_unknowns, n_unknowns))
corr_matrix = np.zeros((n_unknowns, n_unknowns))

# ITERATIONS TO UPDATE PARAMETERS
eps = 1e-6  # epsilon for numerical differentiation, part of forming the A matrix
converged = False   # flag to check if algorithm converged
j_iterations = int(1e4)  # max number of iterations
H = np.zeros(n_unknowns)  # Threshold array (for all parameters)

for j in range(0, j_iterations):
    print('\n----------Loop :', j, '----------')

    # CALCULATE LINEARIZED OBSERVATION EQUATIONS
    dl = np.concatenate((D_obs - model(X[0], X[1], X[2], X[3], X[4]), X_prior - X))

    df_dP = (model(X[0] + eps, X[1], X[2], X[3], X[4]) -
             model(X[0] - eps, X[1], X[2], X[3], X[4])) / (2 * eps)

    df_dT = (model(X[0], X[1] + eps, X[2], X[3], X[4]) -
             model(X[0], X[1] - eps, X[2], X[3], X[4])) / (2 * eps)

    df_dU = (model(X[0], X[1], X[2] + eps, X[3], X[4]) -
             model(X[0], X[1], X[2] - eps, X[3], X[4])) / (2 * eps)

    df_dC = (model(X[0], X[1], X[2], X[3] + eps, X[4]) -
             model(X[0], X[1], X[2], X[3] - eps, X[4])) / (2 * eps)

    df_dD = (model(X[0], X[1], X[2], X[3], X[4] + eps) -
             model(X[0], X[1], X[2], X[3], X[4] - eps)) / (2 * eps)

    A[:n_lambda, 0] = df_dP.reshape(-1)
    A[:n_lambda, 1] = df_dT.reshape(-1)
    A[:n_lambda, 2] = df_dU.reshape(-1)
    A[:n_lambda, 3] = df_dC.reshape(-1)
    A[:n_lambda, 4] = df_dD.reshape(-1)

    # UPDATE PARAMETERS
    N = A.T @ P @ A
    Q = np.linalg.inv(N)
    dX = Q @ A.T @ P @ dl
    X = X + dX
    V = A @ dX - dl

    variance_factor_est = (V.T @ P @ V) / (n_dim - n_unknowns)
    variance_factor_est = variance_factor_est.reshape(-1)[0]  # taking the value from [[val]]
    SIGMA_XX = Q * variance_factor  # multiplying by variance factor = unit variance; not variance_factor_est

    # PREPARE NEXT ITERATION
    for i in range(n_unknowns):
        H[i] = dX[i] / (SIGMA_XX[i, i] ** 0.5)
    # Terminate iterations if all parameters change by less than 1% of their std deviation
    # if j < j_iterations and max(np.abs(H)) < 0.01:
    #     converged = True
    #     break

    if np.abs(dX[i] < 1e-11):
        converged = True
        break

    print(X)


# PRINT FINAL ESTIMATION RESULT
if converged:
    print("Solution Converged")
else:
    print("No Convergence")

print('\nEstimated Parameters: \n', X)
print('\nResiduals', np.linalg.norm(dl[:n_unknowns]))  # check this
# print('\nResiduals', np.linalg.norm(V[:n_unknowns]))  # check this

# GLOBAL QUALITY CHECK
deg_freedom = n_lambda
quality_factor = deg_freedom * (variance_factor_est / variance_factor)
limits = [chi2.ppf(0.0015, df=deg_freedom), chi2.ppf(0.9985, df=deg_freedom)]
print(f'\nQuality Factor: {quality_factor:0.2f}; Range: {limits[0]:0.2f} to {limits[1]:0.2f}')
if limits[0] <= quality_factor <= limits[1]:
    print("QF within Range")
else:
    print("QF outside Range")


# CORRELATION MATRIX
for c_i in range(n_unknowns):
    for c_j in range(n_unknowns):
        corr_matrix[c_i][c_j] = SIGMA_XX[c_i][c_j] / np.sqrt(SIGMA_XX[c_i][c_i] * SIGMA_XX[c_j][c_j])
        corr_matrix[c_i][c_j] = np.round(corr_matrix[c_i][c_j], 2)
# corr_matrix = np.tril(corr_matrix, -1)  # lower triangular matrix

param_names = ['P', 'T', 'U', 'C', 'D']
df_corr_matrix = pd.DataFrame(data=corr_matrix, index=param_names, columns=param_names)
print('\nCorrelation Matrix\n', df_corr_matrix)


# RESULTS
P_est, T_est, U_est, C_est, D_est = X[0], X[1], X[2], X[3], X[4]
std_P_est, std_T_est = SIGMA_XX[0, 0]**0.5, SIGMA_XX[1, 1]**0.5
std_U_est, std_C_est = SIGMA_XX[2, 2]**0.5, SIGMA_XX[3, 3]**0.5
std_D_est = SIGMA_XX[4, 4]**0.5

# CHI2 TEST
delta_X = X.reshape(-1) - X_true.reshape(-1)
delta_X = delta_X.reshape(-1, 1)
global_test = delta_X.T @ np.linalg.inv(SIGMA_XX) @ delta_X / np.linalg.matrix_rank(SIGMA_XX)
if global_test[0][0] <= chi2.ppf(0.9985, df=np.linalg.matrix_rank(SIGMA_XX)):
    print('Chi2 Test Pass')

print('Error (mm): ', (D_est - D_true)*1e3)

print(D_est)

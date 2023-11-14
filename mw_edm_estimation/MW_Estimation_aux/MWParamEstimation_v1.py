import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import CiddorPy
from sklearn.metrics import mean_squared_error

FONT_SIZE = 14
FIG_SIZE = (6, 6)
rc('font', size=FONT_SIZE)
rc('axes', titlesize=FONT_SIZE)


def generate_obs_data(nSamples, true_d):
    temperature = 20
    pressure = 1013.25
    humidity = 50
    conCO2 = 450
    wavelength = np.linspace(550, 1050, nSamples, endpoint=True)
    wavenumber = 1 / (1e-3 * wavelength)

    n_obs = CiddorPy.CiddorHillGroup(wavenumber, pressure, temperature, humidity, conCO2)

    d = true_d * np.ones(nSamples)
    d_obs = n_obs * d
    return n_obs, d_obs, wavelength, wavenumber


n_samples = 10
true_d = 50
N_OBS, D_OBS, LAMBDA, NU = generate_obs_data(n_samples, true_d)
D_OBS = D_OBS.reshape(n_samples, 1)
N_OBS = N_OBS.reshape(n_samples, 1)


def func_model(T, P, H, XC, D):
    N = CiddorPy.CiddorHillGroup(NU, P, T, H, XC)
    model = N * D
    return model


# Initial Guess
T = 18  # 20
P = 1012.25 # 1013.25
H = 48  # 50
XC = 420  # 450
D = 48  #50

X = np.array([T, P, H, XC, D])
X = X.reshape(len(X), 1)

eps = 1e-5
max_iter = np.int(1e5)
for epoch in range(max_iter):
    t = X[0]
    p = X[1]
    h = X[2]
    xc = X[3]
    d = X[4]

    df_t = np.array([func_model(t + eps, p, h, xc, d), func_model(t - eps, p, h, xc, d)])
    df_dt = (df_t[0] - df_t[1]) / (2 * eps)

    df_p = np.array([func_model(t, p + eps, h, xc, d), func_model(t, p - eps, h, xc, d)])
    df_dp = (df_p[0] - df_p[1]) / (2 * eps)

    df_h = np.array([func_model(t, p, h + eps, xc, d), func_model(t, p, h - eps, xc, d)])
    df_dh = (df_h[0] - df_h[1]) / (2 * eps)

    df_xc = np.array([func_model(t, p, h, xc + eps, d), func_model(t, p, h, xc - eps, d)])
    df_dxc = (df_xc[0] - df_xc[1]) / (2 * eps)

    df_d = np.array([func_model(t, p, h, xc, d+eps), func_model(t, p, h, xc, d-eps)])
    df_dd = (df_d[0] - df_d[1]) / (2 * eps)

    A = np.zeros((n_samples, len(X)))
    A[:, 0] = np.array(df_dt)
    A[:, 1] = np.array(df_dp)
    A[:, 2] = np.array(df_dh)
    A[:, 3] = np.array(df_dxc)
    A[:, 4] = np.array(df_dd)

    D_EST = func_model(t, p, h, xc, d).reshape(n_samples, 1)

    dl = D_OBS - D_EST

    dx = np.dot(np.linalg.pinv(A), dl)

    X = X + dx

    if np.max(np.abs(dx)) < 1e-7:
        print("Converged")
        break

    print(X)

print(X)

# -*- coding: utf-8 -*-
"""

Ciddorpy: Python functions for phase and group refractive index in air

based on Ciddor and Hill, Appl. Opt.38, 1663 (1999)
      and Ciddor, Appl. Opt. 35, 1566 (1996)
      with sign corrected version for the derivative of the Sellmeier Terms
      (Eq.(7))
      
authored by Florian Pollinger 
Physikalisch-Technische Bundesanstalt (PTB), Bundesallee 100, 
   38116 Braunschweig, Germany
   
based in part on Visual Basic routines by Karl Meiners-Hagen
   
contact: florian.pollinger@ptb.de

The author assumes no liability for any application of this code.

written in Python 3.7.4
v4 2020-06-19

The software was developed in the course of the 18SIB01 Geometre project. This 
project has received funding from the EMPIR programme co-financed by the
Participating States and from the European Union’s Horizon 2020 research and 
innovation programme. 

-----
Brief overview of the functions

CiddorHillGroup(sigma_in_invmicrons, p_in_hPa ,  t_in_C,  rh, xc):
     provides the group index of refraction of air calculated following
     Ciddor and Hill, Appl. Opt.38, 1663 (1999)
     corrected for the sign error in Equation (B2)
     
CiddorPhase(sigma_in_invmicrons, p_in_hPa ,  t_in_C,  rh , xc):
    provides the phase index of refraction of air calculated following
    Ciddor, Appl. Opt. 35, 1566 (1996)
    
CiddorUncertaintyNIST(sigma_in_invmicrons,  p_in_hPa, t_in_C, rh ):
    provides an estimate for the expanded (k=2) measurement 
    uncertainty of the Ciddor air refractive phase index formula
    as introduced by Jack A. Stone and Jay H. Zimmerman (NIST) 
    in their Engineering Metrology Toolbox
    https://emtoolbox.nist.gov/Wavelength/Documentation.asp#AppendixAV
    
    Additional (usually dominating) uncertainty contributions due to the 
    sensor systems are not included in this value!

UngroupEstimate(sigma_in_invmicrons, p_in_hPa, t_in_C , rh, xc, eps=1e-5):
    provides an estimate for the expanded (k=2) measurement 
    uncertainty of the Ciddor-Hill air group refractive index
    based on the NIST approximation formula for the expanded uncertainty
    of the Ciddor formula for the air refractive index and Ciddor's
    discussion of the measurement uncertainty in Sec. 8 of 
    Appl. Opt. 35, 1566 (1996)
    
    Additional (usually dominating) uncertainty contributions due to the 
    sensor systems are not included in this value!
    
    This function needs CiddorHillGroup(), CiddorPhase(), and 
    CiddorUncertaintyNIST().
"""

import numpy as np

def CiddorHillGroup(sigma_in_invmicrons, p_in_hPa ,  t_in_C,  rh, xc, \
                    svp_over_water = True):
    # Formula based on Ciddor and Hill, Appl. Opt.38, 1663 (1999)
    # and Ciddor, Appl. Opt. 35, 1566 (1996)
    #
    # sigma - wave number in microns^(-1)
    # p_in_hPa - ambient pressure in hPa
    # t_in_C - temperature in °C
    # rh - relative humidity in %r.h.
    # xc - CO2 contents in ppm
    #
    # svp_over_water:
    #   True: saturation vapor pressure over water
    #   False: saturation vapor pressure over ice

    sigma = sigma_in_invmicrons

    #pressure in Pa
    p = p_in_hPa * 100
    
    # Temperatur in K
    tk = t_in_C + 273.15
    # Temperatur in °C
    t = t_in_C

    if svp_over_water:
        # saturation vapor pressure water in Pa: svp
        # Appendix A of Ciddor 1996: SVP A, B, C, D
        AA = 0.000012378847 # K^(-2)
        BB = -0.019121316 # 1/K
        CC = 33.93711047
        DD = -6343.1645 # K
        svp = np.exp(AA * tk * tk + BB * tk + CC + DD / tk)
    else:
        # saturation vapor pressure over ice (Ciddor, 1996, Eq (13))
        exp_ice = -2633.5/tk + 12.537
        svp = pow(10,exp_ice)

    # 'enhancement' factor
    alpha = 1.00062
    beta = 0.0000000314 # /Pa
    gamma = 0.00000056 # °C^-2

    f = alpha + beta * p + gamma * t * t

    # relative humidity
    h = rh / 100

    # water vapor fraction xw 
    xw = f * h * svp / p

    # index of refraction standard atmosphere
    k0 = 238.0185 # \mu m^-2
    k1 = 5792105 # \mu m^-2
    k2 = 57.362 # \mu m^-2
    k3 = 167917 # \mu m^-2

    # eqn[1] Ciddor 1996
    nas = (k1 / (k0 - sigma**2) + k3 / (k2 - sigma**2)) * 0.00000001
    # eqn[2] Ciddor 1996
    naxs = nas * (1 + 0.000000534 * (xc - 450))

    # index of refraction water vapor:
    # cf value below Eqn(3) and w from Appendix A; Ciddor 1996
    cf = 1.022
    w0 = 295.235 # \mu m^-2
    w1 = 2.6422 # \mu m^-2
    w2 = -0.03238 # \mu m^-4
    w3 = 0.004028 # \mu m^-6

    # eqn[3] Ciddor 1996
    nws = cf * (w0 + w1 * sigma**2 + w2 * sigma**4 + w3 * sigma**6) \
    * 0.00000001

    # Ma - molecular mass dry air
    # Below Eqn(4) Ciddor 1996
    Ma = 0.001 * (28.9635 + 0.000012011 * (xc - 400))

    # Za - compressibility standard dry air
    # values provided in  appendix A Ciddor 1996
    a0 = 0.00000158123 # K/Pa
    a1 = -0.000000029331 # /Pa
    a2 = 0.00000000011043 # /K/Pa
    b0 = 0.000005707 # K/Pa
    b1 = -0.00000002051 # /Pa
    c0 = 0.00019898 # K/Pa
    c1 = -0.000002376 # /Pa
    d = 0.0000000000183 # K^2/Pa
    e = -0.00000000765 # K^2/Pa^2

    tkref = 288.15 # K
    tref = tkref - 273.15
    pref = 101325 # Pa
    xwref = 0 #

    Za = 1 - pref / tkref * (a0 + a1 * tref + a2 * tref**2 + (b0 + b1 * tref) \
                             * xwref + (c0 + c1 * tref) * xwref**2)
    Za = Za + (pref / tkref)**2 * (d + e * xwref**2)

    # Zw - compressibility pure water vapor
    tkref = 293.15 # K
    tref = tkref - 273.15
    pref = 1333 # Pa
    xwref = 1 #

    Zw = 1 - pref / tkref * (a0 + a1 * tref + a2 * tref**2 + (b0 + b1 * tref) \
                             * xwref + (c0 + c1 * tref) * xwref**2)
    Zw = Zw + (pref / tkref)**2 * (d + e * xwref**2)


    # rhoaxs - density of standard air
    # Mw - molar mass pure water vapor
    Mw = 0.018015 # kg/mol
    # gas constant:
    R = 8.31451 # J/mol/K

    xwref = 0
    pref = 101325
    tkref = 288.15
    rhoaxs = (pref * Ma / (Za * R * tkref)) * (1 - xwref * (1 - Mw / Ma))


    # rhows density of standard water vapor
    xwref = 1
    pref = 1333
    tkref = 293.15
    rhows = (pref * Ma / (Zw * R * tkref)) * (1 - xwref * (1 - Mw / Ma))


    # actual compressibility (t,p, relative humidity, ...)
    Z = 1 - p / tk * (a0 + a1 * t + a2 * t**2 + (b0 + b1 * t) * xw + \
                      (c0 + c1 * t) * xw**2)
    Z = Z + (p / tk)**2 * (d + e * xw**2)

    # density of dry air
    rhoa = p * Ma * (1 - xw) / Z / R / tk


    # density of water vapor
    rhow = p * Mw * xw / Z / R / tk


    # phase index of refraction Ciddor, 1996
    nprop = rhoa / rhoaxs * naxs + rhow / rhows * nws

    #beginning of group refractive index calculation according
    # to Ciddor and Hill, 1999
    nph = nprop + 1

    #Derivative index of refraction standard atmosphere (modified (B2))
    dnas = (2) * sigma * ((k1 / ((k0 - sigma**2) * (k0 - sigma**2))+ \
                          k3 / ((k2 - sigma**2) * (k2 - sigma**2)))) *\
                          0.00000001 
    dnaxs = dnas * (1 + 0.000000534 * (xc - 450))

    #derivative index of refraction of water vapor
    dnws = 2 * cf * (w1 * sigma + 2 * w2 * sigma**3 + 3 * w3 * sigma**5) \
    * 0.00000001

    #Combination using Lorentz-Lorenz relation (Eq (7) of 
    #Ciddor und Hill, 1999)
    prefac  = sigma * (nph * nph + 2) * (nph * nph + 2) / nph
    naxs += 1
    nws += 1
    SumAXS = ((naxs / (naxs * naxs + 2)) / (naxs * naxs + 2)) * rhoa / rhoaxs \
    * dnaxs
    SumW = ((nws / (nws * nws + 2)) / (nws * nws + 2)) * rhow / rhows * dnws
    ng = nph + prefac * (SumAXS + SumW)
    
    return ng 
    
def CiddorPhase(sigma_in_invmicrons, p_in_hPa ,  t_in_C,  rh , xc, \
                  svp_over_water = True):
    # Formula based on Ciddor, Appl. Opt. 35, 1566 (1996)
    #
    # sigma_in_invmicrons - wavenumber in microns^(-1)
    # p_in_hPa - ambient pressure in hPa
    # t_in_C - temperature in °C
    # rh - relative humidity in %r.h.
    # xc - CO2 contents in ppm
    #
    # svp_over_water:
    #   True: saturation vapor pressure over water
    #   False: saturation vapor pressure over ice

    #pressure in Pa
    p = p_in_hPa * 100
    
    sigma = sigma_in_invmicrons
    
    # Temperatur in K
    tk = t_in_C + 273.15
    # Temperatur in °C
    t = t_in_C

    if svp_over_water:
        # saturation vapor pressure water in Pa: svp
        AA = 0.000012378847 # K^(-2)
        BB = -0.019121316 # 1/K
        CC = 33.93711047
        DD = -6343.1645 # K
        svp = np.exp(AA * tk * tk + BB * tk + CC + DD / tk)
    else:
        # saturation vapor pressure over ice (Ciddor, 1996, Eq (13))
        exp_ice = -2633.5/tk + 12.537
        svp = pow(10,exp_ice)

    # 'enhancement' factor
    alpha = 1.00062
    beta = 0.0000000314 # /Pa
    gamma = 0.00000056 # °C^-2

    f = alpha + beta * p + gamma * t * t

    # relative humidity
    h = rh / 100

    # water vapor fraction xw 
    xw = f * h * svp / p

    # index of refraction standard atmosphere
    k0 = 238.0185 # \mu m^-2
    k1 = 5792105 # \mu m^-2
    k2 = 57.362 # \mu m^-2
    k3 = 167917 # \mu m^-2

    nas = (k1 / (k0 - sigma**2) + k3 / (k2 - sigma**2)) * 0.00000001
    naxs = nas * (1 + 0.000000534 * (xc - 450))

    # index of refraction water vapor:
    cf = 1.022
    w0 = 295.235 # \mu m^-2
    w1 = 2.6422 # \mu m^-2
    w2 = -0.03238 # \mu m^-4
    w3 = 0.004028 # \mu m^-6

    nws = cf * (w0 + w1 * sigma**2 + w2 * sigma**4 + w3 * sigma**6) \
    * 0.00000001

    # Ma - molecular mass dry air
    Ma = 0.001 * (28.9635 + 0.000012011 * (xc - 400))

    # Za - compressibility standard dry air
    a0 = 0.00000158123 # K/Pa
    a1 = -0.000000029331 # /Pa
    a2 = 0.00000000011043 # /K/Pa
    b0 = 0.000005707 # K/Pa
    b1 = -0.00000002051 # /Pa
    c0 = 0.00019898 # K/Pa
    c1 = -0.000002376 # /Pa
    d = 0.0000000000183 # K^2/Pa
    e = -0.00000000765 # K^2/Pa^2

    tkref = 288.15 # K
    tref = tkref - 273.15
    pref = 101325 # Pa
    xwref = 0 #

    Za = 1 - pref / tkref * (a0 + a1 * tref + a2 * tref**2 + (b0 + b1 * tref) \
                             * xwref + (c0 + c1 * tref) * xwref**2)
    Za = Za + (pref / tkref)**2 * (d + e * xwref**2)

    # Zw - compressibility pure water vapor
    tkref = 293.15 # K
    tref = tkref - 273.15
    pref = 1333 # Pa
    xwref = 1 #

    Zw = 1 - pref / tkref * (a0 + a1 * tref + a2 * tref**2 + (b0 + b1 * tref) \
                             * xwref + (c0 + c1 * tref) * xwref**2)
    Zw = Zw + (pref / tkref)**2 * (d + e * xwref**2)


    # rhoaxs - density of standard air
    # Mw - molar mass pure water vapor
    Mw = 0.018015 # kg/mol
    # gas constant:
    R = 8.31451 # J/mol/K

    xwref = 0
    pref = 101325
    tkref = 288.15
    rhoaxs = (pref * Ma / (Za * R * tkref)) * (1 - xwref * (1 - Mw / Ma))


    # rhows density of standard water vapor
    xwref = 1
    pref = 1333
    tkref = 293.15
    rhows = (pref * Ma / (Zw * R * tkref)) * (1 - xwref * (1 - Mw / Ma))


    # actual compressibility (t,p, relative humidity, ...)
    Z = 1 - p / tk * (a0 + a1 * t + a2 * t**2 + (b0 + b1 * t) * xw + \
                      (c0 + c1 * t) * xw**2)
    Z = Z + (p / tk)**2 * (d + e * xw**2)

    # density of dry air
    rhoa = p * Ma * (1 - xw) / Z / R / tk


    # density of water vapor
    rhow = p * Mw * xw / Z / R / tk


    # phase index of refraction Ciddor, 1996
    return  1 + rhoa / rhoaxs * naxs + rhow / rhows * nws
    
def CiddorUncertaintyNIST(sigma_in_invmicrons,  p_in_hPa, t_in_C, rh):
    # =========================================================================
    # Uncertainty estimate according to NIST 
    # (Jack A. Stone and Jay H. Zimmerman)
    # Engineering Metrology Toolbox
    # https://emtoolbox.nist.gov/Wavelength/Documentation.asp#AppendixAV
    # 
    # Equation A50 for the phase refractivity uncertainty  
    # 
    # covers only estimate of k=2 uncertainty of formula
    # sensor uncertainties must be separately taken into account
    #
    # wavenumber sigma in microns^(-1)
    # temperature t_in_C in °C
    # ambient pressure p_in_hPa in Pa
    # relative humidity rh in % rel. humidity
    # =========================================================================

    sigma = sigma_in_invmicrons

    p = p_in_hPa*100 #Pa
    t = t_in_C

    # Temperatur in K
    tk = t + 273.15
    
    # relative humidity in %rh:
    h = rh / 100

    # saturation pressure water in Pa: svp
    AA = 0.000012378847#
    BB = -0.019121316#
    CC = 33.93711047#
    DD = -6343.1645#
    svp = np.exp(AA * tk * tk + BB * tk + CC + DD / tk)

    # "Enhancement" factor:
    alpha = 1.00062
    beta = 0.0000000314 # /Pa
    gamma = 0.00000056 # °C^-2

    f = alpha + beta * p + gamma * t * t

    pv = f * h * svp
    
    fac1 = 0.003*p/(t+273)
    fac2 = 4 + 6*pow(10,-8)*(p-10**5)*(p-10**5)+0.006*(t-20)*(t-20)
    sum1 = fac1 * fac1 * fac2
    sum2 = pv * pv * ((8*pow(10,-4))**2 +(6*pow(10,-6)*(sigma**4)) *\
                      (6*pow(10,-6)*(sigma**4)))
    UCidNIST = pow(10,-8)*np.sqrt(sum1 + sum2)
    
    return UCidNIST

def UngroupEstimate(sigma_in_invmicrons, p_in_hPa, t_in_C , rh, xc, eps=1e-5):
    # sigma_in_invmicrons and eps - wavenumbers in inverse microns
    # p_in_hPa - pressure in hPa
    # t_in_C - temperature in °C
    # feuchte in % rel.
    # xc - CO2 contents in ppm
    #
    # estimating the uncertainty of the group index formula
    # following Ciddor's approach to estimate sensitivity of 
    # group refractive index to uncertainty
    #
    # using NIST estimate of the phase formulae uncertainty
    #
    # result k=2
    
    sigma0 = sigma_in_invmicrons
    
    p = p_in_hPa 
    temp = t_in_C
    feuchte = rh
    
    sigmaplus = sigma0+eps
    sigmaminus = sigma0 - eps
    
    # derivative dn_group/dn_Ph approximated
    # by (dn_group/dsigma)/(dn_Ph/dsigma)
    # according to Ciddor 1996, Sec. 8
    
    nphplus = CiddorPhase(sigmaplus, p, temp, feuchte, xc)
    nphminus = CiddorPhase(sigmaminus, p, temp, feuchte, xc)
    dnPhdsigma = (nphplus - nphminus) / (2 * eps)

    ngplus = CiddorHillGroup(sigmaplus, p, temp, feuchte, xc)
    ngminus = CiddorHillGroup(sigmaminus, p, temp, feuchte, xc)
    dngdsigma = (ngplus - ngminus)/(2*eps)
    
    dngdnph = np.abs(dngdsigma / dnPhdsigma)
    
    UngEstimate =dngdnph * CiddorUncertaintyNIST(sigma0, p, temp, feuchte)

    return UngEstimate





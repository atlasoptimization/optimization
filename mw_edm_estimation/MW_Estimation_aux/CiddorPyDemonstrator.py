# -*- coding: utf-8 -*-
"""
authored by Florian Pollinger 
Physikalisch-Technische Bundesanstalt (PTB), Bundesallee 100, 
   38116 Braunschweig, Germany

This program provides a simple text-based calculation interface
demonstrating the capabilities of the CiddorPy functions.
contact: florian.pollinger@ptb.de

The author assumes no liability for any application of this code.

The software was developed in the course of the 18SIB01 Geometre project. This 
project has received funding from the EMPIR programme co-financed by the
Participating States and from the European Union’s Horizon 2020 research and 
innovation programme. 

"""

import numpy as np
import CiddorPy as cid
    
l0=633
sigma0 = 1 / (l0 / 1000)
p0 = 1013.25
t0 = 20
rh0= 50
CO20=450
svp_over_water_0 = True

EnvCond = np.array([0., 0., 0., 0., 0.], dtype = float) 
lInd = 0
tInd = 1
pInd = 2
rhInd = 4
xcInd = 3
IndMax = 4

svp_select = 7
stop = 9

EnvCond[lInd]= l0
EnvCond[pInd]= p0
EnvCond[tInd]= t0
EnvCond[rhInd]= rh0
EnvCond[xcInd] = CO20
proz = '%rh'

#limits from the 'Engineering Metrology Toolbox', NIST 
# by Jack A. Stone and Jay H. Zimmerman
#https://emtoolbox.nist.gov/Wavelength/Ciddor.asp

limits=np.arange(10, dtype = float).reshape(5,2)
limits[lInd] = [300,1700]
limits[tInd] = [-40,100]
limits[pInd] = [100, 1400]
limits[rhInd] = [0,100]
limits[xcInd] = [0, 2000] 


running = bool(True)

print()
print('CiddorPy')
print('Calculation of the air phase index of refraction according to'\
              ' Ciddor, 1996')
print('and of the air group index of refraction according to'\
              ' Ciddor and Hill, 1996,')
print('corrected for a sign error')   

while running:
    try:
        print()
        print('%d'%lInd,' - wavelength in nm' , '(%.0f' %limits[lInd,0], \
            'to %.0f' %limits[lInd,1],') : %.3f' %EnvCond[lInd])
        print('%d'%tInd,' - temperature in °C' , '(%.0f' %limits[tInd,0], \
            'to %.0f' %limits[tInd,1],'): %.3f' %EnvCond[tInd])
        print('%d'%pInd,' - pressure in hPa' , '(%.0f' %limits[pInd,0], \
            'to %.0f' %limits[pInd,1],'): %.5f' %EnvCond[pInd])
        print('%d'%xcInd,' - carbon dioxide in ppm' , '(%.0f' %limits[lInd,0],\
            'to %.0f' %limits[lInd,1],'): %.3f' %EnvCond[xcInd])
        print('%d'%rhInd,' - relative humidity in %s'%proz, '(%.0f' \
            %limits[rhInd,0], 'to %.0f' %limits[rhInd,1],'): %.3f' \
            %EnvCond[rhInd])
        if svp_over_water_0:
            print('%d' %svp_select,' - standard vapor pressure over water')
        else:
            print('%d' %svp_select,' - standard vapor pressure over ice')
        print()
        nPh = cid.CiddorPhase(1000/EnvCond[lInd],EnvCond[pInd],EnvCond[tInd],\
                          EnvCond[rhInd],EnvCond[xcInd], svp_over_water_0)
        UnPh = cid.CiddorUncertaintyNIST(1000/EnvCond[lInd],EnvCond[pInd],\
                                     EnvCond[tInd],EnvCond[rhInd])
        print('n_phase: %.9f' %nPh)
        print('U(n_phase)_formula: %.9f'%UnPh)
        ng = cid.CiddorHillGroup(1000/EnvCond[lInd],EnvCond[pInd],EnvCond[tInd], \
                             EnvCond[rhInd],EnvCond[xcInd] , svp_over_water_0)
        Ung=cid.UngroupEstimate(1000/EnvCond[lInd],EnvCond[pInd],EnvCond[tInd], \
                             EnvCond[rhInd],EnvCond[xcInd])
        print()
        print('n_group: %.9f' %ng)
        print('U(n_group)_formula: %.9f'%Ung)
        print()
        print('%d'%stop,' - leave program')
        print()
        print('All uncertainties are k=2 estimates for the uncertainty of', \
              'the formulae only')
        a = input("Please select: ")
        n=int(a)
        if n == stop:
            running = bool(False)
        else:
            if n == 7:
               svp_over_water_0 = not(svp_over_water_0)
            else:
                if (n>-1) and (n < IndMax+1):
                    val = float(input('New value: '))
                    if ((val<limits[n,0]) or (val>limits[n,1])):
                        print('Value exceeds recommended range!')
                    else:
                        EnvCond[n]=float(val)
                else:
                    print('Invalid selection, please try again!')   
    except: 
        print('Invalid entry, please try again!')
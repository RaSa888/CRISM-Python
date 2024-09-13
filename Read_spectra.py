#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:01:07 2020

@author: pragya
"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from scipy.optimize import curve_fit as cf
from scipy import constants
from scipy.fftpack import fft 
from scipy.fftpack import ifft 


#%%
'''
<<<USGS SPECLIB 7 Resampled to CRISM Targeted>>>
'''


chl=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/s07CRSMj_Chlorite_HS179.2B_ASDFRb_AREF.txt', skiprows=1)
kies=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/s07CRSMj_Kieserite_KIEDE1.a_crse_gr_ASDFRc_AREF.txt', skiprows=1)
nont=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/s07CRSMj_Nontronite_NG-1.a_ASDNGb_AREF.txt', skiprows=1)
sap=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/s07CRSMj_Saponite_SapCa-1.AcB_BECKb_AREF.txt', skiprows=1)
ser=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/s07CRSMj_Serpentine_HS318.3B_ASDFRc_AREF.txt', skiprows=1)


co2_1=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/1.txt', skiprows=3)
co2_2=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/2.txt', skiprows=3)



wvl=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/s07CRSMj_CRISM_Waves_JOINED_MTR3_microns_489ch.txt', skiprows=1)

chl[np.where(chl==-1.2300000e+34)]=0 
kies[np.where(kies==-1.2300000e+34)]=0
nont[np.where(nont==-1.2300000e+34)]=0
sap[np.where(sap==-1.2300000e+34)]=0
ser[np.where(ser==-1.2300000e+34)]=0
co2_1[np.where(co2_1>10)]=0
co2_2[np.where(co2_2>100)]=0

#%%
"""CO2 SPECTRA"""
# index=np.linspace(0,437,438)
plt.plot(co2_1[:,1])
plt.show()
# plt.xticks(index, np.round(wvl[0:437],2), rotation=70)
plt.plot(co2_2[:,1])
plt.show()


plt.plot(abs(np.fft.fft(co2_1[260:310,1])**2/25)[1:25])
plt.plot(abs(np.fft.fft(co2_2[260:310,1])**2/25)[1:25], 'g')
plt.show()

plt.plot(abs(np.fft.fft(co2_1[:,1])**2/219)[1:50])
plt.show()
plt.plot(abs(np.fft.fft(co2_2[:,1])**2/219)[1:50], 'g')
plt.show()

print(wvl[260],wvl[310])

#%%

'''
plt.plot(wvl, chl)
plt.plot(wvl, kies)
plt.plot(wvl, nont)
plt.plot(wvl, sap)
plt.plot(wvl, ser)
'''
# plt.plot(wvl[0:299], chl[0:299]**3*kies[0:299], label='chl')

# plt.plot(wvl[0:299], kies[0:299], label='kies')
# plt.plot(wvl[0:299], chl[0:299])


F_chl=fft(chl[0:299])
F_kies=fft(kies[0:299])
F_nont=fft(nont[0:299])


# plt.plot(abs(F_chl[1:20]))
# plt.plot(abs(F_kies[1:20]))
# plt.plot(abs(F_nont[1:20]))


a=(chl[0:299]+kies[0:299]+nont[0:299])
# plt.plot(wvl[0:299], a[0:299], label='Chl+Kies')


F_chl_kies=fft(a)

F_diff=F_chl_kies-F_chl-F_nont/4
kies_inv=ifft(F_diff)

F_diff=F_chl_kies-F_kies-F_nont/4
chl_inv=ifft(F_diff)

F_diff=F_chl_kies-F_chl-F_kies/4
nont_inv=ifft(F_diff)


plt.subplot(3,1,1)
plt.plot(wvl[0:299], kies[0:299], label='kies')
plt.plot(wvl[0:299], kies_inv[0:299], label='Kies_inv')
plt.legend()

plt.subplot(3,1,2)
plt.plot(wvl[0:299], chl[0:299], label='chl')
plt.plot(wvl[0:299], chl_inv[0:299], label='CHL_inv')
plt.legend()

plt.subplot(3,1,3)
plt.plot(wvl[0:299], nont[0:299], label='nont')
plt.plot(wvl[0:299], nont_inv[0:299], label='Nont_inv')
plt.legend()

#%%
image=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.img')
#%%
from pandas import read_csv
img = read_csv('Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.img')
#%%
f = open("Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.img", "r")





#%%
from scipy.spatial import ConvexHull, convex_hull_plot_2d
hull=ConvexHull(chl)
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
    
    
    



















#%%
'''
<<<OLD CODE USING RELAB spectra:: DONT USE>>>


cho=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/RelabDatabase2019Dec31/data/cho/ll/cgn197.txt',  delimiter='\t', skiprows=2)
qt=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/RelabDatabase2019Dec31/data/jat/qt/bua2qt033.txt',  delimiter='\t', skiprows=2)

c_wl=[]
crf=[] #red shift


for i in range(len(cho)):
    c_wl.append(cho[i][0])
    crf.append(cho[i][1])
    

c_rf=np.array(crf)

plt.plot(c_wl,c_rf)
plt.plot(c_wl,0.01/c_rf)

#


q_wl=[]
q_rf=[] #red shift


for i in range(len(qt)):
    q_wl.append(qt[i][0])
    q_rf.append(qt[i][1])
    

#plt.plot(q_wl,q_rf)

#%%

f=fft(rf)
plt.stem(abs(f[1:200])/400*2)
'''
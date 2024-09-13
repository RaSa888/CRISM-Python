#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:01:00 2020

@author: Ranjan Sarkar
"""

import numpy as np
from spectral import *
import matplotlib.pyplot as plt
#%%
#GET IMAGE
L1 = envi.open('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.hdr', '/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.img')
DDR = envi.open('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_de180l_ddr1.hdr','/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_de180l_ddr1.img')

#GET WAVELENGTHS
wvl=np.genfromtxt('/Users/pragya/Desktop/Spectroscopy_ML/HRL/CRISM_wavl_file.csv', delimiter=',', invalid_raise = False)
wvl=wvl.flatten()
wvl =wvl[~np.isnan(wvl)] #very interesting Check!!! ::: x = x[numpy.logical_not(numpy.isnan(x))] and wvl[np.isfinite(wvl)] and shortcut: wvl[~np.isnan(wvl)]

#COPY IMAGE TO ARRAY FOR ANALYSIS
a=np.array(L1.load())
a[np.where(a>1)]=0 #Replace all 65535 values with zero.

#%%
# SOME PLOTTING
# plt.plot(wvl, a[34,55,:])
# plt.show()
co2=np.where((1.8<=wvl) & (wvl<=2.2))#Band indices for co2.
co2_i=co2[0][:] # np.where returned a tuple, it is difficult to index with a tuple.
# plt.plot(wvl[co2_i], a[134, 235, co2_i])
# plt.show()
#%%
# LOAD CO2 Specctra
co2_1=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/1.txt', skiprows=3)
co2_2=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/2.txt', skiprows=3)

co2_1[np.where(co2_1>10)]=0
co2_2[np.where(co2_2>100)]=0
co2_1=co2_1[::-1]

#%%
w1=np.where(wvl==1.98084)[0] # index of wavelength 1
w1=np.int(w1)
w2=np.where(wvl==2.00723)[0]
w2=np.int(w2)
deno=np.log((co2_1[w1,1])/(co2_1[w2,1])) # denominator
beta1=np.zeros((390,320)) 
nume=np.zeros((390,320)) # numerator
for i in range(390):
    for j in range(320):
        if np.round(a[i,j,w2],3)!=0.0:
            nume[i,j]=np.log(a[i,j,w1])/np.log(a[i,j,w2])
            beta1[i,j]=nume[i,j]/deno
        else:
            beta1[i,j]=0
        

co2_1_scaled=np.zeros(438)

#%%
i,j=0,0
L1_corr=np.zeros((390,320,438))
for i in range(0,390):
    for j in range(0,320):
        L1_corr[i,j,:]=a[i,j,:]/np.cos(np.deg2rad(DDR[i,j,5]))/np.power(co2_1[:,1],beta1[i,j])
plt.plot(wvl, L1_corr[34,75,:], label='Corr--')
plt.plot(wvl, a[34,75,:], label='Uncorr--')
plt.plot(wvl, np.power(co2_1[:,1],beta1[34,75]))
plt.legend()
plt.show()

#%%
# m=np.max(a[23,74,:])
# plt.plot(wvl, a[23,74,:])
# plt.plot(wvl, a[23,74,:]/m)
# # plt.plot(wvl, co2_1[:,1])


#%%
# B_cor_a=np.zeros((390,320,438))
# for i in range(390):
#     for j in range(320):
#         B_corr_a[i,j]=a[i,j,:]*DDR[i,j]*
    
#%%
import pysptools.spectro as spectro


cr=spectro.convex_hull_removal(a[34,75,:], wvl)
plt.plot(wvl,cr[0])

plt.plot(wvl,a[34,75,:])

plt.plot(wvl,L1_corr[34,75,:], label='Corr--')
plt.legend()
plt.show()
#%%
import scipy.signal as sg
sgf=sg.savgol_filter(a[34,75,:],15,5,deriv=1)
plt.plot(wvl,sgf)
# plt.plot(a[34,75,:])
#%%


L1_CR=np.zeros(390,320,438)
for i in range(0,390):
    for j in range(0,320):
        cr=spectro.convex_hull_removal(a[i,j,:], wvl)
        L1_CR[i,j,:]=cr[0]


s1=a[:,:,w1-127]
s2=a[:,:,w2+57]

plt.scatter(s1,s2, s=1)


































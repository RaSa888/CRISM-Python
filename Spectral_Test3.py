#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:14:11 2020

@author: Ranjan
"""
import numpy as np
from spectral import *
import matplotlib.pyplot as plt
import pysptools.spectro as spectro
import scipy.signal as sg

#%%
# GET IMAGE
L1 = envi.open('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.hdr', '/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.img')
DDR = envi.open('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_de180l_ddr1.hdr','/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_de180l_ddr1.img')

# GET WAVELENGTHS
wvl=np.genfromtxt('/Users/pragya/Desktop/Spectroscopy_ML/HRL/CRISM_wavl_file.csv', delimiter=',', invalid_raise = False)
wvl=wvl.flatten()
wvl =wvl[~np.isnan(wvl)] #very interesting Check!!! ::: x = x[numpy.logical_not(numpy.isnan(x))] and wvl[np.isfinite(wvl)] and shortcut: wvl[~np.isnan(wvl)]

# COPY IMAGE TO ARRAY FOR ANALYSIS
a=np.array(L1.load())
a[np.where(a>1)]=0 #Replace all 65535 values with zero.

# LOAD CO2 Specctra
co2_1=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/1.txt', skiprows=3)
co2_2=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/2.txt', skiprows=3)

co2_1[np.where(co2_1>10)]=0
co2_2[np.where(co2_2>10)]=0

co2_1=co2_1[::-1]

#%%
# Clip image to bands from 2:248
img=a[:,:,2:248]
wvl=wvl[2:248]
co2_1=co2_1[2:248][1]

#%%
# Set Wavelengths

co2=np.where((1.8<=wvl) & (wvl<=2.2))#Band indices for co2.
co2_i=co2[0][:] # np.where returned a tuple, it is difficult to index with a tuple.
plt.plot(wvl[co2_i], img[134, 235, co2_i])

w1=np.where(wvl==1.98084)[0] # index of wavelength 1
w1=np.int(w1)
w2=np.where(wvl==2.00723)[0]
w2=np.int(w2)

#%%
# Continuum Removal -----TAKES A LOT OF TIME TO RUN THIS CELL, DIRECTLY LOAD THE SAVED RESULT (SEE THE CELL AFTER THE NEXT)
img_cr=np.zeros((390,320,246))
for i in range(0,390):
    for j in range(0,320):
        a=img[i,j,:]
        if np.mean(a)!=0:
            cr=spectro.convex_hull_removal(a, wvl)
            img_cr[i,j,:]=cr[0]
        else:
            img_cr[i,j,:]=np.linspace(0,0,246)
#%%
# Sgolay the spectrum 
img_sg=np.zeros((390,320,246))
for i in range(0,390):
    for j in range(0,320):
        sgf=sg.savgol_filter(img_cr[i,j,:],15,5,deriv=0)
        img_sg[i,j,:]=sgf

plt.plot(wvl,img_cr[34,65,:],label="Continuum Removed")        
plt.plot(wvl,img_sg[34,65,:], label="SGolay")
plt.plot(wvl,img[34,65,:],label="Original Spectrum")
plt.legend()
plt.show()
# plt.plot(a[34,75,:])
#%%
# Save the processed images
envi.save_image('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT_Continuum_Removed.hdr', img_cr, ext='bil', interleave = 'BIL', force=True)
envi.save_image('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT_Continuum_Removed_SGol.hdr', img_sg, ext='bil', interleave = 'BIL')




#%%
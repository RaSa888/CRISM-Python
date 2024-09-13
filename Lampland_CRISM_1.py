#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 08:41:09 2020

@author: pragya
"""

import numpy as np
from spectral import *
import matplotlib.pyplot as plt
import pysptools.spectro as spectro
import scipy.signal as sg

#%%
# GET IMAGE
#L1 = envi.open('/Volumes/256GB/Lampland_CRISM/hrl0000641c_07_if180l_trr3_CAT_corr_p-ENVI-Std.hdr')


#L1 = envi.open('/Volumes/256GB/Lampland_CRISM/hrl0000641c_07_if180j_mtr3.hdr')
#L1 = envi.open('/Volumes/256GB/Lampland_CRISM/hrl00005ae4_07_if180j_mtr3.hdr')
L1 = envi.open('/Volumes/256GB/Lampland_CRISM/hrl00007a4f_07_if180j_mtr3.hdr')
               


# GET WAVELENGTHS
wvl=np.genfromtxt('/Users/pragya/Desktop/Spectroscopy_ML/HRL/CRISM_wavl_file.csv', delimiter=',', invalid_raise = False)
wvl=wvl.flatten()
wvl =wvl[~np.isnan(wvl)] #very interesting Check!!! ::: x = x[numpy.logical_not(numpy.isnan(x))] and wvl[np.isfinite(wvl)] and shortcut: wvl[~np.isnan(wvl)]

# COPY IMAGE TO ARRAY FOR ANALYSIS
img=np.array(L1.load())
img[np.where(img>1)]=0 #Replace all 65535 values with zero.

# Clip image to bands from 2:248
img2=img[:,:,3:247]
img=img2
img2=0
wvl=wvl[3:247]


# # AND CLIP IMAGE TO DROP THE FIRST AND LAST BANDS
# img2=img[:,:,1:-1]
# img=img2

rows=img.shape[0]
cols=img.shape[1]
bands=img.shape[2]


# Mean centering the data


f=np.reshape(img,(rows*cols, bands))  # Unfolding the data. Vector to hold all pixels in one row
mean_sp=np.mean(f, axis=0)  # Calculation of mean spectra 
f_ma=f-mean_sp # Mean adjusted (For multispectral data use autoscaling)


#%%
# DISPLAY THE IMAGE. PLOT SPECTRA BY DOUBLE CLICKING

#view1=imshow(img, bands=(32,54,75))
#%% 
""" THINGS I NEED TO LEARN 
PCA; MNF; ICA; Least Squares (Linear & Non-linear); PPI, n-D visualization; Spectral unmixing

"""

#%% MULTIPLICATIVE SCATTER CORRECTION (Code modified from:  https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/)

ref_sp=np.mean(f_ma, axis=0) # Reference spectrum for MSC

f_MSC=np.zeros_like(f_ma)

for i in range(f_ma.shape[0]-1):
    fit=np.polyfit(ref_sp, f_ma[i,:], 1, full=True)
    f_MSC[i,:]=(f_ma[i,:]-fit[0][1])/fit[0][0]

print("Done...MSC")

#%% 
# PCA

#simg=img[150:300,120:240,:] #simg subset of the image
# pc=principal_components(simg)
# xdata = pc.transform(simg)
#w = view_nd(xdata[:,:,:15]) Doesn't work, throws back error!


f_ma=f_MSC # Comment out this line if you don't want to use MSC


cv_matrix=np.dot(np.transpose(f_ma),f_ma)
w,v=np.linalg.eig(cv_matrix)

p=10 # User selected number of principal components to include

reduced_img=np.dot(np.transpose(v[:,0:p-1]),np.transpose(f_ma)) # Reduced data
Final_image=np.dot(v[:,0:p-1],reduced_img) # Retrive the original dimension reduced image
Final_image=np.reshape(np.transpose(Final_image),(rows,cols,bands)) # Convert back to the rows X cols form
#view2=imshow(Final_image, bands=(32,54,75))

# CREATE Biplots; Too many dta points, plotting will take some time!

PC1=np.dot(f,v[:,0]) 
PC1.shape
PC2=np.dot(f,v[:,1])
PC3=np.dot(f,v[:,2])
plt.scatter(PC1,PC2, s=1)
plt.scatter(PC2,PC3, s=1)
plt.scatter(PC1,PC3, s=1)

#%%
# MNF

f_mnf=np.reshape(Final_image,(rows*cols, bands))
mean_sp_mnf=np.mean(f_mnf, axis=0)
f_ma_mnf=f-mean_sp_mnf
cv_matrix_mnf=np.dot(np.transpose(f_ma_mnf),f_ma_mnf)
w_mnf,v_mnf=np.linalg.eig(cv_matrix_mnf)

p_mnf=3 # User selected number of principal components to include

reduced_img_mnf=np.dot(np.transpose(v_mnf[:,0:p-1]),np.transpose(f_ma_mnf)) # Reduced data
Final_image_mnf=np.dot(v_mnf[:,0:p-1],reduced_img_mnf) # Retrive the original dimension reduced image
Final_image_mnf=np.reshape(np.transpose(Final_image_mnf),(rows,cols,bands)) # Convert back to the rows X cols form
view3=imshow(Final_image_mnf, bands=(32,54,75))









































 
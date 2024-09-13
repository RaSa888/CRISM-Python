#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 00:55:58 2020

@author: pragya
"""
import numpy as np
from spectral import *
import matplotlib.pyplot as plt
import pysptools.spectro as spectro
import scipy.signal as sg

#%%
# GET IMAGE
L1 = envi.open('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT_Continuum_Removed.hdr', '//Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT_Continuum_Removed.bil')

# GET WAVELENGTHS
wvl=np.genfromtxt('/Users/pragya/Desktop/Spectroscopy_ML/HRL/CRISM_wavl_file.csv', delimiter=',', invalid_raise = False)
wvl=wvl.flatten()
wvl =wvl[~np.isnan(wvl)] #very interesting Check!!! ::: x = x[numpy.logical_not(numpy.isnan(x))] and wvl[np.isfinite(wvl)] and shortcut: wvl[~np.isnan(wvl)]

# COPY IMAGE TO ARRAY FOR ANALYSIS
img=np.array(L1.load())
img[np.where(img>1)]=0 #Replace all 65535 values with zero.

# LOAD CO2 Specctra
co2_1=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/1.txt', skiprows=3)
co2_2=np.loadtxt('/Users/pragya/Desktop/Spectroscopy_ML/USGS_SPECLIB_CRISM_TARGET/2.txt', skiprows=3)

co2_1=co2_1[::-1,1]

#%%



# Clip image to bands from 2:248
wvl=wvl[3:247]
co2_1=co2_1[3:247]

# Continuum removal for CO2
co2_1_cr=spectro.convex_hull_removal(co2_1, wvl)   #continuum removal for CO2 
co2cr=np.array(co2_1_cr[0]) # Have to copy the continuum removed co2 spectrum from a tuple into the array.

# AND CLIP IMAGE TO DROP THE FIRST AND LAST BANDS
img2=img[:,:,1:-1]
img=img2



#%%




#%%
# Sgolay the spectrum 
# img_sg=np.zeros((390,320,246))
# for i in range(0,390):
#     for j in range(0,320):
#         sgf=sg.savgol_filter(img_cr[i,j,:],15,5,deriv=0)
#         img_sg[i,j,:]=sgf

# plt.plot(wvl,img_cr[34,65,:],label="Continuum Removed")        
# plt.plot(wvl,img_sg[34,65,:], label="SGolay")
# plt.plot(wvl,img[34,65,:],label="Original Spectrum")
# plt.legend()
# plt.show()
# # plt.plot(a[34,75,:])






#%%
# Set Wavelengths

co2=np.where((1.8<=wvl) & (wvl<=2.2))#Band indices for co2.
co2_i=co2[0][:] # np.where returned a tuple, it is difficult to index with a tuple.
plt.plot(wvl[co2_i], img[134, 235, co2_i])
plt.show()

w1=np.where(wvl==1.98084)[0] # index of wavelength 1
w1=np.int(w1)
w2=np.where(wvl==2.00723)[0]
w2=np.int(w2)
#%%
#Scaling the CO2 spectrum to match the scene spectrum, McGuire et al., 2009 >>>> VOLCANO SCAN

deno=np.log((co2cr[w1])/(co2cr[w2])) # denominator
beta=np.zeros((390,320)) 
nume=np.zeros((390,320)) # numerator
for i in range(390):
    for j in range(320):
        if np.round(img[i,j,w2],3)!=0.0:
            nume[i,j]=np.log(img[i,j,w1])/np.log(img[i,j,w2])
            beta[i,j]=nume[i,j]/deno
        else:
            beta[i,j]=0
        

co2cr_scaled=np.zeros(244)
img_corr=np.zeros((390,320,244))
for i in range(0,390):
    for j in range(0,320):
        if np.round(beta[i,j],3)!=0.0:
            co2cr_scaled=np.power(co2cr,beta[i,j])
            img_corr[i,j,:]=img[i,j,:]/co2cr_scaled



#%%
#Plot a scatter diagram
rw1=img[:,:,w1]
rw2=img[:,:,w2]
plt.scatter(rw1,rw2, s=1)
plt.show()
#%%
#Normal FFT

# x=img[124,125,:]
# plt.plot(abs(np.fft.fft(x))[1:122]*2/244)

# plt.show()

#%%
#Perform a Short-Time Fast Fourier Transform in the spectral domain

nume1=np.log(img[257,137,w1])/np.log(img[257,137,w2])
beta1=nume1/deno           
co2cr_scaled1=np.power(co2cr, beta1)




f_c, t_c, Zxx_c = sg.stft(co2cr_scaled1[co2_i], nperseg=15) #STFT of CO2


x=img[257,137,:] #Sample spectrum

f, t, Zxx = sg.stft(x[co2_i], nperseg=15)
plt.pcolormesh(t, f, np.abs(Zxx), vmin=0.0015, vmax=0.008,  shading='gouraud')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


#%%
#Inverse STFFT


Z_diff=Zxx-Zxx_c #CO2 removal in the spectral domain


# plt.pcolormesh(t, f, np.abs(Z_diff), vmin=0.0015, vmax=0.008,  shading='gouraud')



# plt.show()


# plt.pcolormesh(t, f, np.abs(Zxx), vmin=0.0015, vmax=0.008,  shading='gouraud')
# plt.show()



# plt.pcolormesh(t, f, np.abs(Zxx_c), vmin=0.0015, vmax=0.008,  shading='gouraud')


# plt.show()



tm, ix=sg.istft(Zxx_c, nperseg=15)

plt.plot(ix)
plt.show()




#%%
# plt.plot(wvl, co2cr, label='CO2')
# plt.plot(wvl, co2cr_scaled1, label='CO2_scaled')

plt.plot(wvl, img[343,265,:], label='Scene Spectra')

plt.plot(wvl, img_corr[343,265,:], label='Scene Spectra VS')


plt.legend()
plt.show()





#%%
# plt.plot(co2cr_scaled)
plt.plot(co2cr_scaled1)
# plt.plot(co2cr)
plt.show()

#%% ONLY CO2 BAND
def err(scene, co2):
    X=0
    for i in range(61):
        X+=(scene[i]-co2[i])**2
    return np.sqrt(X)

    


row=43
col=65
plt.figure(figsize=(12,16))
for i in range(1,13):
    plt.subplot(4,3,i)
    beta=(i/12)
    plt.plot(wvl[co2_i],img[row,col,co2_i]/np.power(co2cr[co2_i],beta), label='Corr')
    plt.plot(wvl[co2_i],img[row,col,co2_i],label='Scene')
    plt.plot(wvl[co2_i],np.power(co2cr[co2_i],beta),label='$CO_2$')
    error=np.round(err(img[row,col,co2_i],np.power(co2cr[co2_i],beta)), 3)
    plt.axvline(x=wvl[w1], color='k', linewidth=0.5)
    plt.axvline(x=wvl[w2], color='k', linewidth=0.5)
    title='beta='+ np.str(np.round(beta,2))+ '; row='+np.str(row)+'; col='+np.str(col)+ '; Err='+np.str(error)
    plt.title(title)
    
plt.legend()   
plt.tight_layout(pad=3)
plt.show()
#%%  WHOLE IMAGE
row=43
col=65
plt.figure(figsize=(12,16))
for i in range(1,13):
    plt.subplot(4,3,i)
    beta=(i/12)
    plt.plot(wvl,img[row,col,:]/np.power(co2cr,beta), label='Corr')
    # plt.plot(wvl,img[row,col,:],label='Scene')
    # plt.plot(wvl,np.power(co2cr,beta),label='$CO_2$')
    error=np.round(err(img[row,col,:],np.power(co2cr,beta)), 3)
    plt.axvline(x=wvl[w1], color='k', linewidth=0.5)
    plt.axvline(x=wvl[w2], color='k', linewidth=0.5)
    title='beta='+ np.str(np.round(beta,2))+ '; row='+np.str(row)+'; col='+np.str(col)+ '; Err='+np.str(error)
    plt.title(title)
    
plt.legend()   
plt.tight_layout(pad=3)
plt.show()

#%% DERIVATIVE SPECTRA
row=164
col=264
a=img[row,col,:]

plt.figure(figsize=(12,16))
for i in range (1,7):
    plt.subplot(3,2,i)
    plt.plot(np.diff(a[co2_i],n=i),label=np.str(i)+'scene')
    plt.plot(np.diff(co2cr[co2_i],n=i), label='co2')
    plt.legend()

plt.show()




#%%
# #%%  PCA  <<<< https://www.neonscience.org/classification-pca-python >>>>

# # make a copy of the original (continuum removed) image
# I=img[1:388,17:313,:].copy()


# # calculate mean spectrum of the whole image and subtract it from the original image
# Z=np.sum(I,axis=0)
# Z2=np.sum(Z, axis=0)
# mean_sp=Z2/(387*296)

# I-=mean_sp

# # Calculate the covariance matrix
# T = I.reshape(387*296, 244).copy()
# T=np.transpose(T)
# NSamps   = np.shape(T)[1]
# C=np.cov(T)

# w,v=np.linalg.eig(C)


# TPCA = np.dot(v.T, T)

# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure(1)
# ax = Axes3D(fig)
# ax.scatter(TPCA[0,range(NSamps)],TPCA[1,range(NSamps)],TPCA[2,range(NSamps)], marker='o')
# plt.show()

# for coord in range(3):
#     P1 = TPCA[coord, :]
#     PCAIm      = np.reshape(P1, (387, 296))
#     plt.figure(2+coord)
#     plt.imshow(np.abs(PCAIm))
#     plt.colorbar()
#     plt.show()

# #%%
# pc=principal_components(I)


# pc_0999 = pc.reduce(fraction=0.999)

# img_pc = pc_0999.transform(img)
#%%








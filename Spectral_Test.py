#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 13:57:47 2020

@author: pragya
"""

import numpy as np
from spectral import *
import matplotlib.pyplot as plt

L1 = envi.open('/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.hdr', '/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.img')
# L2 = envi.open(/Users/pragya/Desktop/Spectroscopy_ML/HRL/hrl00005ae4_07_if180l_trr3_CAT.hdr)



#%%
wvl=np.genfromtxt('/Users/pragya/Desktop/Spectroscopy_ML/HRL/CRISM_wavl_file.csv', delimiter=',', invalid_raise = False)
wvl=wvl.flatten()
wvl =wvl[~np.isnan(wvl)] #very interesting Check!!! ::: x = x[numpy.logical_not(numpy.isnan(x))] and wvl[np.isfinite(wvl)] and shortcut: wvl[~np.isnan(wvl)]

print(wvl.shape)
#%%
# spectral.settings.WX_GL_DEPTH_SIZE = 16
# view_cube(a,bands =[30,50,60]) # WILL WORK ONLY WHEN RUN WITH python.app
# plt.pause(10)
#%%
# ImageView(L1, (29, 65))
# view = imshow(img, (29, 19, 9))
a=np.array(L1.load())
a[np.where(a>1)]=0
#%%
# view=imshow (a, (76,5,349), cmap='hot')
view1=imshow(a[:, :, 323])#, cmap='Accent')

#%%

(m, c) = kmeans(a, 4, 6)
#%%
imshow(m, cmap='Accent') 

plt.figure()
for i in range (c.shape[0]):
    plt.plot(wvl,c[i])
plt.grid()
plt.show()
    

#%%
view = imshow(a[:,:,324], classes=m)

view.set_display_mode('overlay')

view.class_alpha = 0.1
plt.show()
#%%
plt.plot(wvl, a[34,55,:])
plt.show()


#%%
# %matplotlib qt
co2=np.where((1.8<=wvl) & (wvl<=2.2))
co2_i=co2[0][:] # np.where returned a tuple, it is difficult to index with a tuple.
plt.plot(wvl[co2_i], a[134, 235, co2_i])
# plt.ioff()
plt.show()
#%%
view1=imshow(a, bands=(32,54,75)) #, cmap='Accent') # ONLY THIS WAY YOU CAN PLOT SPECTRA BY DOUBLE CLICKING

# view1=imshow(a, (32,54,75)) #, cmap='Accent') # THIS WILL WORK TOO!!

# imshow(L1, bands=(43,54,23))# THIS WILL WORK TOO!! But have to get rid of the 65535 values!!

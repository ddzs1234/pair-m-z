#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:43:18 2020

@author: ashley
"""

"""
compare dap fitting results and pPXF fitting results
1. continuum
2. simultaneously continuum and emission
"""

import numpy as np
import os
import sys
sys.path.insert(0, '/home/ashley/zs/daily_code/')
from read_map_cube import read_map_cube
import matplotlib.pyplot as plt
from astropy.table import Table


def map_cube(plateifu):
    wave,flux,ivar,flux_header,redden,flux_map,ivar_map,mask_map,ellcoo,stellar_vel,\
        flux_map_header=read_map_cube(plateifu)

    x_center=np.int(flux_header['CRPIX1'])-1
    y_center = np.int(flux_header['CRPIX2']) - 1
    
    nii=flux_map[19][x_center,y_center]
    oiii=flux_map[13][x_center,y_center]
    halpha=flux_map[18][x_center,y_center]
    hbeta=flux_map[11][x_center,y_center]
    
    dir1='/media/ashley/project/pair_galaxy/2020-1-31/bpt/fitting/map_cube_compare/'
    with open(dir1+'%s.txt'%plateifu,'a+') as f:
        print('nii map ',nii,file=f)
        print('halpha map ',halpha,file=f)
        print('oiii map ',oiii,file=f)
        print('hbeta map ',hbeta,file=f)
    
    t1=np.transpose([wave,flux[x_center, y_center]*redden,np.sqrt(redden**2/ivar[x_center,y_center])])
    t=Table(t1,names=['wave','flux','flux_error'],dtype=['f8','f8','f8'])
    if not os.path.exists(dir1+'%s_gas_rotation_dered.fits'%plateifu):
        t.write(dir1+'%s_gas_rotation_dered.fits'%plateifu)
    
if __name__=='__main__':
    map_cube('10001-12702')
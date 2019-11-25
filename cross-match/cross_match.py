import os
import sys
from time import time

import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from collections import OrderedDict

"""
1. dr7 mpl8 690 arcsec cross match -> gal_info_v2.fits
2. 1output file 于mpl8进行进一步的match
"""


def crossmatch(X1, X2, K,max_distance=np.inf):

    X1 = np.asarray(X1, dtype=float)
    X2 = np.asarray(X2, dtype=float)

    N1, D = X1.shape
    N2, D2 = X2.shape

    if D != D2:
        raise ValueError('Arrays must have the same second dimension')

    kdt = cKDTree(X2)
    
    print(X1)
    dist, ind = kdt.query(X1,k=K, distance_upper_bound=max_distance)

    return dist, ind

def crossmatch_angular(X1, X2, K,max_distance):
    
    X1 = X1 * (np.pi / 180.)
    X2 = X2 * (np.pi / 180.)
    max_distance = max_distance * (np.pi / 180.)

    # Convert 2D RA/DEC to 3D cartesian coordinates
    Y1 = np.transpose(np.vstack([np.cos(X1[0]) * np.cos(X1[1]),
                                 np.sin(X1[0]) * np.cos(X1[1]),
                                 np.sin(X1[1])]))
    Y2 = np.transpose(np.vstack([np.cos(X2[:, 0]) * np.cos(X2[:, 1]),
                                 np.sin(X2[:, 0]) * np.cos(X2[:, 1]),
                                 np.sin(X2[:, 1])]))

    # law of cosines to compute 3D distance
    max_y = np.sqrt(2 - 2 * np.cos(max_distance))
    dist, ind = crossmatch(Y1, Y2,K, max_y)

    # convert distances back to angles using the law of tangents
    not_inf = ~np.isinf(dist)
    x = 0.5 * dist[not_inf]
    dist[not_inf] = (180. / np.pi * 2 * np.arctan2(x,
                                  np.sqrt(np.maximum(0, 1 - x ** 2))))

    return dist, ind

def r2arcsec(r):
    return r * 180 / np.pi

class cmatch(object):
    """
    """
    def __init__(self,plateifu,manga_ra,manga_dec,manga_z,dr7,dr7_z,dir_res,dir_1):
        
        self.plateifu=plateifu
        self.m_ra=manga_ra
        self.m_dec=manga_dec
        self.m_z=manga_z
        self.dr7=dr7
        self.dr7_z=dr7_z
        self.dir_res=dir_res
        self.dir_1=dir_1

        
        self.max_dis_deg()      
        self.dis_vel_cut()
        
        
        
    def max_dis_deg(self):
        '''
        150kpc ~= degree
        '''
        dis = cosmo.comoving_distance(self.m_z)  # Mpc
        r = 0.15 / dis.value
        self.max_r = r2arcsec(r)  # degree
        return self.max_r
        
    def dis_vel_cut(self):
        """
        150kpc,1000km/s
        """
        manga=np.array([self.m_ra,self.m_dec],dtype=np.float64)
        print(self.plateifu)
        dist, ind = crossmatch_angular(manga, self.dr7, self.max_r,10)
        match = ~np.isinf(dist)
        dist_match = dist[match]
        ind_match=ind[match]

        if len(dist_match)>0:
            if len(dist_match)==10:
                with open(self.dir_1,'a+') as f_10:
                    print(self.plateifu,self.m_ra,self.m_dec,self.max_r,self.m_z,file=f_10)
            elif len(dist_match)<10:
                with open(self.dir_res+self.plateifu+'_dis_vel_match.txt','a+') as f_res:
                    print('index','ra','dec','manga_ra','manga_dec','min_dis_deg','max_r','manga_z','dr7_z',file=f_res)
                    for j in range(0,len(ind_match)):
                        ra=self.dr7[:,0][ind_match[j]]
                        dec=self.dr7[:,1][ind_match[j]]
                        z=self.dr7_z[ind_match[j]]
                        if abs(self.m_z-z)<(1/300):
                            print(ind_match[j],ra,dec,self.m_ra,self.m_dec,dist_match[j],self.max_r,self.m_z,z,file=f_res)
                        else:
                            with open(self.dir_res+self.plateifu+'_dis_match.txt','a+') as f_res1:
                                print(ind_match[j],ra,dec,self.m_ra,self.m_dec,dist_match[j],self.max_r,self.m_z,z,file=f_res1)
        else:
            with open(self.dir_res+'no_match.txt','a+') as f_no_match:
                print(self.plateifu,file=f_no_match)
        

        
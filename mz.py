import os
import random

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table,vstack,hstack
from PyAstronomy.pyasl import binningx0dt
from scipy import interpolate, stats
from cycler import cycler
import matplotlib.colors as mcolors

plt.rcParams["axes.prop_cycle"] = cycler(color=mcolors.TABLEAU_COLORS)



def pair_delta(pair_row, f_z, f_z_e):
    """
    pair_row:
    0:mass_p 2:z_p 4:z_e_p 6:ifu_p
    1:mass_s 3:z_s 5:z_e_s 7:ifu_s
    """
    # real delta_z:
    delta_z = float(pair_row[2]) - float(pair_row[3])
    delta_z_e = np.sqrt(float(pair_row[4])**2 + float(pair_row[5])**2)
    # mzr predict:
    delta_z_mzr = f_z(float(pair_row[0])) - f_z(float(pair_row[1]))
    delta_z_e_mzr = np.sqrt(f_z_e(float(pair_row[0]))**2 + f_z_e(float(pair_row[1]))**2)
    # return real then mzr predict
    return delta_z, delta_z_e, delta_z_mzr, delta_z_e_mzr

def special_ifu(array,how,ifu_p,ifu_s,mass_p,mass_s):
    """
    return ifu of special sources:
    """
    if how=='min':
        index=np.where(array==np.min(array))[0]
#         print(index)
#         print('min','ifu_p','ifu_s','mass_p','mass_s')
#         print(ifu_p[index],ifu_s[index],mass_p[index],mass_s[index])
    elif how=='max':
        index=np.where(array==np.max(array))[0]
#         print(index)
#         print('max','ifu_p','ifu_s','mass_p','mass_s')
#         print(ifu_p[index],ifu_s[index],mass_p[index],mass_s[index])
        
        
def ratio(pair):
    """
    add str ratio
    """
    mask_p = (abs(pair[:,8]) < abs(pair[:,10]))
    # the first bin:
    mask1 = (pair[:,10] < 0.2)
    mask1_1p = mask1 & mask_p
    if pair[:,10][mask1].size==0:
        ratio1_p=sigma1_p=0
        ratio1_p_s=str(0)
    else:
        ratio1_p=pair[:,10][mask1_1p].size/pair[:,10][mask1].size
        sigma1_p=ratio1_p*(1-ratio1_p)
        ratio1_p_s=str(pair[:,10][mask1_1p].size)+'/'+str(pair[:,10][mask1].size)

    # the second bin:
    mask2 = (pair[:,10] >= 0.2) & (pair[:,10] <= 0.4)
    mask2_1p = mask2 & mask_p
    if pair[:,10][mask2].size==0:
        ratio2_p=sigma2_p=0
        ratio2_p_s=str(0)
    else:
        ratio2_p=pair[:,10][mask2_1p].size/pair[:,10][mask2].size
        sigma2_p=ratio2_p*(1-ratio2_p)
        ratio2_p_s=str(pair[:,10][mask2_1p].size)+'/'+str(pair[:,10][mask2].size)

    # the third bin:
    mask3 = (pair[:,10] > 0.4)
    mask3_1p = mask3 & mask_p
    if pair[:,10][mask3].size==0:
        ratio3_p=sigma3_p=0
        ratio3_p_s=str(0)
    else:
        ratio3_p=pair[:,10][mask3_1p].size/pair[:,10][mask3].size
        sigma3_p=ratio3_p*(1-ratio3_p)
        ratio3_p_s=str(pair[:,10][mask3_1p].size)+'/'+str(pair[:,10][mask3].size)
    
    return [ratio1_p,sigma1_p,ratio2_p,sigma2_p,ratio3_p,sigma3_p],[ratio1_p_s,ratio2_p_s,ratio3_p_s]
           
class mzr(object):
    """
    mzr purpose:
    ============
    """

    def __init__(self,
                 plateifu,
                 z,
                 z_e,
                 indicator_name,
                 mass,
                 mass_name,
                 dis,
                 dvel,
                 time,
                 dir1,
                 pairdir,
                 plot=True,
                 save=False,
                 errorbar=False):
     
        self.plateifu = plateifu
        self.z = z.astype(np.float)
        self.z_e = z_e.astype(np.float)
        self.indicator_name = indicator_name
        self.mass = mass.astype(np.float)
        self.mass_name = mass_name
        self.dir1 = dir1
        self.pairdir = pairdir
        self.dis = dis
        self.dvel = dvel
        self.time = time
        self.save=save
        self.plot=plot
        self.errorbar=errorbar
        
        self.no_effect_point()
        self.remove_nouse_data()
        self.MZR()
        self.mzfunc()
        self.pair()
        self.group()
        self.control_sample()
#         self.distribution()
        self.pick_ctsamp_same_mass_pair()
        self.plot_delta_z()
        self.bin_delta_z_mzr()
        self.plot_ratio()
        
        
    def no_effect_point(self):
        """
        find no effective points
        """
        mask = (self.z > 9.5) | (self.z < 8.0) | (self.mass < 9.0)
        # mass limit for manga 10^9 Msun
        mass_noeff = self.mass[mask]
        z_noeff = self.z[mask]
        ifu_noeff = self.plateifu[mask]
        z_e_noeff = self.z_e[mask]
        names = ['plateifu', 'mass', 'z', 'z_e']
        if self.save:
            t = Table([ifu_noeff, mass_noeff, z_noeff, z_e_noeff], names=names)
            t.write(self.dir1 + '%s_strange_%s.fits' %
                    (self.indicator_name, self.mass_name))

    def remove_nouse_data(self):
        """
        remove wrong data
        """
        length = len(self.mass)
        mask = (self.mass > 8.0) & (self.z > 0)
        a_mass=self.mass
        a_z=self.z
        self.mass = self.mass[mask]
        self.z = self.z[mask]
        self.z_e = self.z_e[mask]
        # plateifu 
        a_ifu=self.plateifu
        self.plateifu = self.plateifu[mask]
        mask1 = (a_mass <= 8.0) | (a_z <= 0)
        
        assert len(self.mass) == len(self.z) == len(self.z_e) == len(
            self.plateifu
        ), "after remove no use data, mass and z and z_e and plateifu must have the same length"
        assert len(self.mass) + \
            len(a_ifu[mask1]) == length, "miss some point"
        if self.save:
            t=Table([a_ifu[mask1],a_mass[mask1],a_z[mask1]],names=['plateifu','mass','z'])
            t.write(self.dir1+'remove_%s_%s.fits'%(self.indicator_name,self.mass_name))

    def MZR(self):
        """
        plot mzr relation
        """
        # 分成15个bin

        self.median_z, edge_z, num_z = stats.binned_statistic(self.mass,
                                                              self.z,
                                                              'median',
                                                              bins=15)
        self.std_z, _, _ = stats.binned_statistic(self.mass,
                                                  self.z,
                                                  'std',
                                                  bins=15)
        self.binx = []
        for i in range(0, len(edge_z) - 1):
            self.binx.append((edge_z[i] + edge_z[i + 1]) / 2)

        if self.plot:
            plt.figure()
            plt.scatter(self.mass, self.z, s=3, color='orange')
            plt.errorbar(self.binx, self.median_z, yerr=self.std_z, fmt='rp--')
            plt.xlabel('$Log(M/M_{\odot})$')
            plt.ylabel('$12+log(O/H)$')
            plt.xlim(8.0,1.02*np.max(self.mass))
            plt.ylim(7.9,9.8)
            plt.title('%s_%s : 1-1.5re' %
                      (self.indicator_name, self.mass_name))
            if self.save:
                plt.savefig(self.dir1 + '%s_%s_mzr.jpg' %
                            (self.indicator_name, self.mass_name),
                            dpi=300)
            else:
                plt.show()
            plt.clf()

    def mzfunc(self):
        """
        mzr function: with mass, we can obtain ideal z;
        """
        self.f_z = interpolate.interp1d(self.binx,
                                        self.median_z,
                                        fill_value='extrapolate')
        self.f_z_e = interpolate.interp1d(self.binx,
                                          self.std_z,
                                          fill_value='extrapolate')

    def pair(self):
        """
        pair galaxy information
        mask sure: M_primary > M_secondary
        self.pair:
        0:pri_mass 2.pri_z 4.pri_z_e 6.pri_ifu 
        1.sec_mass 3.sec_z 5.sec_z_e 7.sec_ifu 
        8.delta_z 9.delta_z_e
        10.delta_z_mzr 11.delta_z_mzr
        """
        
        self.pair_mass = []
        self.pair_z = []

        f_pair = fits.open(self.pairdir + '%s_pair_mpl8_%s_%s.fits' %
                           (self.time, self.dis, self.dvel))
        pifu = f_pair[1].data.field('primary_ifu')
        sifu = f_pair[1].data.field('secondary_ifu')
        
        f_pair_but_no_info=open(self.dir1+'pair_but_no_info_%s_%s_%s.txt'%(self.indicator_name,self.dis,self.dvel),'a+')

        # collect pair information
        self.pair = []
        for i in range(0, len(pifu), 1):
            index_p = np.where(self.plateifu == pifu[i])[0]
            index_s = np.where(self.plateifu == sifu[i])[0]
            pair_row = []
            if index_p.size > 0 and index_s.size > 0:
                assert len(index_p) == len(index_s), "primary and secondary must be the same row "
                assert len(
                    index_p
                ) == 1, "more than 1 primary galaxies exist in z_info.fits"
                assert len(
                    index_s
                ) == 1, "more than 1 secondary galaxies exist in z_info.fits"

                # if M_pri < M_sec: exchange
                if self.mass[index_p] < self.mass[index_s]:
                    a = index_p
                    b = index_s
                    index_p = b
                    index_s = a

                pair_row = [
                    self.mass[index_p][0], self.mass[index_s][0], self.z[index_p][0],
                    self.z[index_s][0], self.z_e[index_p][0], self.z_e[index_s][0],
                    self.plateifu[index_p][0], self.plateifu[index_s][0]
                ]  # 大质量，小质量，金属丰度，金属丰度，z_e,z_e,plateifu,plateifu

                delta = pair_delta(pair_row, self.f_z, self.f_z_e)
                for j in delta:
                    pair_row.append(j)
                # now pair_row contain delta information

                self.pair.append(pair_row)

            else:
                print("z_%s doesn't contain pair galaxy information: "%self.indicator_name, pifu[i],
                      sifu[i],file=f_pair_but_no_info)
        self.pair=np.array(self.pair,dtype=object)
        f_pair_but_no_info.close()
        if self.save:
            t = Table(self.pair,
                      names=[
                          'PRI_MASS', 'SEC_MASS', 'PRI_Z', 'SEC_Z', 'PRI_Z_E',
                          'SEC_Z_E', 'PRI_IFU', 'SEC_IFU', 'DELTA_Z',
                          'DELTA_Z_E', 'DELTA_Z_MZR', 'DELTA_Z_E_MZR'
                      ])
            t.write(self.dir1 + 'pair_{0}_{1}_{2}.txt'.format(self.indicator_name,self.dis,self.dvel),format='ascii')

    def group(self):
        """
        is there a group contain lots of galaxies?
        """
        inter_pair = np.intersect1d(self.pair[:,6], self.pair[:,7])
        if len(inter_pair)>0:
            print('there is a group', inter_pair)
        else:
            print('there isnot a group')
    
    def control_sample(self):
        """
        remove pair from whole galaxy sample
        """
        # all pair
        self.conpair_mass = np.concatenate((self.pair[:,0], self.pair[:,1]),axis=None)
        self.conpair_z = np.concatenate((self.pair[:,2], self.pair[:,3]),axis=None)
        self.conpair_z_e = np.concatenate((self.pair[:,4], self.pair[:,5]),axis=None)
        self.conpair_ifu = np.concatenate((self.pair[:,6], self.pair[:,7]),axis=None)
        # remove pair
        self.ctsamp_mass = np.setdiff1d(self.mass, self.conpair_mass)
        self.ctsamp_z = np.setdiff1d(self.z, self.conpair_z)
        self.ctsamp_z_e = np.setdiff1d(self.z_e, self.conpair_z_e)
        self.ctsamp_ifu = np.setdiff1d(self.plateifu, self.conpair_ifu)
        # ? order 应该是一样的

    def distribution(self):
        if self.plot:
            # stellar mass
            plt.figure()
            # x=pair,y=others
            x_num, x_edge = plt.hist(self.conpair_mass, bins=6)
            x1_num, x1_edge = plt.hist(self.pair[0], bins=6)  # primary
            x2_num, x2_edge = plt.hist(self.pair[1], bins=6)  # secondary
            y_num, y_edge = plt.hist(self.ctsamp_mass, bins=6)
            x_edge_middle = []
            x1_edge_middle = []
            x2_edge_middle = []
            y_edge_middle = []
            for i in range(0, len(x_edge) - 1):
                x_edge_middle.append((x_edge[i] + x_edge[i + 1]) / 2)
                x1_edge_middle.append((x1_edge[i] + x2_edge[i + 1]) / 2)
                x2_edge_middle.append((x2_edge[i] + x2_edge[i + 1]) / 2)
                y_edge_middle.append((y_edge[i] + y_edge[i + 1]) / 2)

            plt.scatter(x_edge_middle, x_num, fmt="--", s=4, label='pair')
            plt.scatter(x1_edge_middle, x1_num, fmt='--', s=4, label='primary')
            plt.scatter(x2_edge_middle,
                        x2_num,fmt='--',
                        s=4,
                        label='secondary')
            plt.scatter(y_edge_middle, y_num, fmt='--', s=4, label='others')
            plt.legend()
            plt.xlabel('$Log(M/M_{\odot})$')
            plt.ylabel('$ N $')
            plt.title('stellar mass distribution')
            if save:
                plt.savefig(self.dir1 +
                            '%s_%s_%s_stellar_mass_distribution.jpg' %
                            (self.indicator_name, self.dis, self.dvel),
                            dpi=300)

            # to do SFR
            # to do SMR
            # to do distance：
            # to do 可能有质量一样的源;

    def pick_ctsamp_same_mass_pair(self):
        """
        randomly select galaxies with same stellar mass with pairs
        """
        loop = 150
        each_num = 5
        bin_len = 0.3  # ?
        #float
        self.pair[:,0]=self.pair[:,0].astype(np.float)
        self.conpair_mass=self.conpair_mass.astype(np.float)

        # return index
        index_p = []
        index_s = []
        self.random = []
        self.ratio_random=[]
        c_random_bin=[]
        for i in range(0, loop):
            random_bin=[] # collect each loop of information of random pairs:

            for j in range(0, len(self.pair[:,0])):
                pri = float(self.pair[j,0])
                ind = np.arange(len(self.conpair_mass))
                mask = (self.conpair_mass > (pri - bin_len)) & (self.conpair_mass
                                                              < (pri + bin_len))
                sec = float(self.pair[j,1])
                mask_sec = (self.conpair_mass > (sec - bin_len)) & (
                    self.conpair_mass < (sec + bin_len))
                ind_p = ind[mask]
                ind_s = ind[mask_sec]
                
                ind_p_r = random.choices(ind_p, k=each_num)
                ind_s_r = random.choices(ind_s, k=each_num)
                for k in range(0,len(ind_p_r)):
                    ind_p_r_k=ind_p_r[k]
                    ind_s_r_k=ind_s_r[k]           
                    random_row = np.array([
                    self.conpair_mass[ind_p_r_k], self.conpair_mass[ind_s_r_k],
                    self.conpair_z[ind_p_r_k], self.conpair_z[ind_s_r_k],
                    self.conpair_z_e[ind_p_r_k], self.conpair_z_e[ind_s_r_k],
                    self.conpair_ifu[ind_p_r_k], self.conpair_ifu[ind_s_r_k]
                ],dtype=object)
                    delta = pair_delta(random_row, self.f_z, self.f_z_e)
                    random_row = np.append(random_row, [delta])
                    self.random.append(list(random_row))
                    random_bin.append(list(random_row))
#                     print(random_row[10])
            r,_=ratio(np.array(random_bin,dtype=object))
            c_random_bin.append(random_bin)
            self.ratio_random.append(r)
        self.ratio_random=np.array(self.ratio_random,dtype=object)
        self.random=np.array(self.random,dtype=object)
        
#         t = Table(np.array(c_random_bin,dtype=object),
#           names=[
#               'PRI_MASS', 'SEC_MASS', 'PRI_Z', 'SEC_Z',
#               'PRI_Z_E', 'SEC_Z_E', 'PRI_IFU', 'SEC_IFU',
#               'DELTA_Z', 'DELTA_Z_E', 'DELTA_Z_MZR',
#               'DELTA_Z_E_MZR'
#           ])
#         t.write(self.dir1 + '%s_%s_%s_random.txt' %
#                 (self.indicator_name, self.dis, self.dvel),format='ascii')
        
#         if self.save:
        t1 = Table(self.random,
                  names=[
                      'PRI_MASS', 'SEC_MASS', 'PRI_Z', 'SEC_Z',
                      'PRI_Z_E', 'SEC_Z_E', 'PRI_IFU', 'SEC_IFU',
                      'DELTA_Z', 'DELTA_Z_E', 'DELTA_Z_MZR',
                      'DELTA_Z_E_MZR'
                  ])
        t1.write(self.dir1 + '%s_%s_%s_random_random.txt' %
                (self.indicator_name, self.dis, self.dvel),format='ascii')
        


    def plot_delta_z(self):
        if self.errorbar:
            plt.figure()
            plt.plot(self.random[:,10], self.random[:,8], 'ro', label='random')
            plt.errorbar(self.pair[:,10],
                         self.pair[:,8],
                         fmt='b+',
                         yerr=self.pair[:,9],
                         xerr=self.pair[:,11],                        
                         label='pair')            
            rang_x = np.linspace(np.min(self.pair[:,10]),
                                 1.1 * np.max(self.pair[:,10]), 10)
            plt.plot(rang_x, rang_x, "-.", label='1:1')
            plt.plot(rang_x, -rang_x, '-.', label='1:-1')
            plt.xlim(np.min(self.pair[:,10])-0.01,1.1 * np.max(self.pair[:,10]))
            plt.legend()
        else:
            plt.figure()
#             print('self.random[:,10] ',self.random[:,10])
            plt.scatter(self.random[:,10], self.random[:,8], s=3,c='#8F00FF',label='random')
            plt.scatter(self.pair[:,10], self.pair[:,8], s=4,c='r',label='pair')            
            rang_x = np.linspace(np.min(self.pair[:,10]),
                                 1.1 * np.max(self.pair[:,10]), 10)
            plt.plot(rang_x, rang_x, "-.",color='orange', label='1:1')
            plt.plot(rang_x, -rang_x, '-.',color='orange', label='1:-1')
            plt.xlim(np.min(self.pair[:,10])-0.01,1.1*np.max(self.pair[:,10]))
            plt.legend()
            
        special_ifu(self.pair[:,10],'min',self.pair[:,6],self.pair[:,7],self.pair[:,0],self.pair[:,1])
        plt.show()
        plt.clf()

    def bin_delta_z_mzr(self):
        """
        divide delta_z_mzr into 3 bins
        ？二项分布不知道对不对
        """
        #1
        random_r1=np.mean(self.ratio_random[:,0])
        random_s1=np.mean(self.ratio_random[:,1])
        
        #2
        random_r2=np.mean(self.ratio_random[:,2])
        random_s2=np.mean(self.ratio_random[:,3])
        
        #3
        random_r3=np.mean(self.ratio_random[:,4])
        random_s3=np.mean(self.ratio_random[:,5])
        
        #pair
        ratio_pair,self.ratio_pair_s=ratio(self.pair)
        ratio_pair=np.array(ratio_pair)
        ratio_pair=np.append(ratio_pair,[random_r1,random_s1,random_r2,random_s2,random_r3,random_s3])
        
        self.ratio=np.array(ratio_pair)

    
    def plot_ratio(self):
        """
        ratio figure
        """
        ind=np.arange(3)
        width=0.35
        fig, ax = plt.subplots()
        rect1 = ax.bar(ind-width/2,
                        np.around(self.ratio[0:6:2],decimals=2),
                        width,
                        yerr=self.ratio[1:6:2],
                        color='SkyBlue',
                        label='pair')
        rect2 = ax.bar(ind + width / 2,
                        np.around(self.ratio[6:12:2],decimals=2),
                        width,
                        yerr=self.ratio[7:12:2],
                        color='IndianRed',
                        label='random')
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('ratio between 1:1 and 1:-1')
        ax.set_title('mass %s : %s kpc, %s km/s' %
                     (self.indicator_name, self.dis, self.dvel))
        ax.set_xticks(ind)
        ax.set_xticklabels(
            ('$low \Delta log(O/H)$', '$median \Delta log(O/H)$',
             '$high \Delta log(O/H)$'))
        ax.legend()
        # autolabel 
        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """

            xpos = xpos.lower()  # normalize the case of the parameter
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() * offset[xpos],
                        1.01 * height,
                        '{}'.format(height),
                        ha=ha[xpos],
                        va='bottom')
        def autolabel2(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """

            xpos = xpos.lower()  # normalize the case of the parameter
            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off
            for i in range(0,len(rects)):
                rect=rects[i]
                str_h=self.ratio_pair_s[i]
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() * offset[xpos],
                        1.01 * height,
                        '{}'.format(str_h),
                        ha=ha[xpos],
                        va='bottom')
        autolabel2(rect1, "left")
        autolabel(rect2, "right")
        plt.savefig(self.dir1 + 'ratio_%s_%s_%s_%s_p.jpg' %
                    (self.indicator_name,self.time,self.dis,self.dvel),
                    dpi=300)
        plt.show()
        plt.clf()

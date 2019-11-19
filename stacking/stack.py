import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 
from astropy.table import Table
from scipy.interpolate import interp1d
from matplotlib.pyplot import cm
np.seterr(divide='ignore', invalid='ignore')

class stack(object):
    """
    stacking 0-1re
    S/N>5 sigma
    mask (5570-5590A)
    plot: wavelength_flux,imshow,bpt
    """
    def __init__(self,plateifu,wave,flux,ivar,flux_map,ivar_map,mask,ellcoo,v,dirres,zifu,plot=True):
        self.plateifu=plateifu
        self.flux = flux
        self.wave = wave
        self.ivar=ivar
        self.flux_map=flux_map
        self.ivar_map=ivar_map
        self.mask=mask
        self.ellcoo = ellcoo
        self.v=v
        self.dir=dirres
        self.zifu=zifu
        
        """
        count
        """
        self.pre=self.remove_agn_badspx_ellcoo()
        self.vel2z()
        self.stack()
        self.plot()
        self.min_snr_emission_do2()
        self.min_snr_emission_kd02()
        with open(self.dir+'large_than_0p_3.txt','a+') as f_vaild,\
                open(self.dir+'less_than_0p_3.txt','a+') as f_novalid:
            if self.pre>0.3:
                print(self.plateifu,file=f_vaild)
            else:
                print(self.plateifu,file=f_novalid)
            
            
#             if plot:
             

        
    def remove_agn_badspx_ellcoo(self):
        """
        remove bad spaxel in map data
        """
        mask_bad=(self.mask!=0) # !=0 remove
        self.flux_map=np.ma.array(self.flux_map,mask=mask_bad)
        self.ivar_map=np.ma.array(self.ivar_map,mask=mask_bad)
        
        """
        BPT diagram remove AGN
        Kauffmann 2003: log(OIII/hb)>0.61/(log(NII/Ha)-0.05)+1.3
        [oiii]5008,4960:13,12(5007)
        Hb4862:11
        NII6549,6585:17,19(6584)
        Ha6564:18
        """        
        self.x=np.log10(self.flux_map[19]/self.flux_map[18])
        self.y=np.log10(self.flux_map[13]/self.flux_map[11])
        self.mask_sf=(self.y<0.61/(self.x-0.05)+1.3)&(self.x<0.05)&(self.y<(0.61/(self.x-0.47)+1.19))
        self.mask_agn=(1-self.mask_sf).astype(np.bool)

        self.mask_1=(self.ellcoo<=1)
        num_all=np.sum(self.mask_1)
        self.mask_2=self.mask_1&(self.mask_sf)
        num_valid=np.sum(self.mask_2)
        self.mask_3=self.mask_1&(self.mask_agn)
        
        """
        count effective points
        """
        pre=num_valid/num_all

        """
        drp logcube
        """
        self.flux=self.flux[self.mask_2]
        self.v=self.v[self.mask_2]
        self.ivar=self.ivar[self.mask_2]
        
        return pre
                
        
    
    def vel2z(self):
        self.z=self.v/(3e5)

        
    def stack(self):
        """
        stacking the spectrum
        """
        self.flux_stack=np.zeros(self.wave.shape[0])
        self.error_stack=np.zeros(self.wave.shape[0])
        for i in range(0,len(self.flux),1):
            flux=self.flux[i]
            ivar=self.ivar[i]
            z=self.z[i]
            wave_rest=self.wave/(1+z)
            """
            5570-5590A
            """

            mask_sky=(self.wave<5570)|(self.wave>5590)
            flux1=np.array(flux[mask_sky])
            ivar1=np.array(ivar[mask_sky])
            wave_rest=np.array(wave_rest[mask_sky])
            flux=np.interp(self.wave,wave_rest,flux1)
            ivar=np.interp(self.wave,wave_rest,ivar1)
            # remove ivar==0:
            if flux.shape[0]>0:
                # shape not size
                mask=(ivar==0)
                flux=np.ma.array(flux,mask=mask)
                ivar=np.ma.array(ivar,mask=mask)
                self.flux_stack+=flux
                self.error_stack+=1/ivar
                # ivar=sum(ivar)            
        """
        SNR
        """
        snr=np.sum(self.flux_stack)/np.sqrt(np.sum(self.error_stack))
        table=[self.wave]
        table.append(self.flux_stack)
        table.append(self.error_stack)
        table=np.transpose(table)
        t=Table(table,names=['wave','flux','error'])
        t.write(self.dir+'%s_1re_stack.fits'%self.plateifu,format='fits')
        with open(self.dir+'snr_large_than_5.txt','a+') as f_snr5, open(self.dir+'snr_small_than_5.txt','a+') as f_snrless_5:
            if snr>5:            
                print(self.plateifu,file=f_snr5)
            else:      
                print(self.plateifu,file=f_snrless_5)
            
    def plot(self):
        """
        plot (wave,flux)
        """
        
        plt.figure(figsize=(20,20))
        ax1=plt.subplot(211) # spectrum
        ax2=plt.subplot(245) # imshow
        ax3=plt.subplot(246) # bpt
        ax4=plt.subplot(247)
        ax5=plt.subplot(248)
        
        ax2.imshow(self.mask_1,cmap=cm.Blues) 
        ax2.set_title('region')
        ax3.imshow(self.mask_2,cmap=cm.Oranges)   
        ax3.set_title('sf region')
        ax4.imshow(self.mask_3,cmap=cm.Greys)
        ax4.set_title('no-sf region')

        x_bpt = np.arange(np.min(self.x)-0.01, 0.07, 0.02)
        y_bpt = 0.61/(x_bpt-0.47)+1.19
        x_bpt1 = np.arange(np.min(self.x)-0.01, -0.3, 0.02)
        y_bpt1 = 0.61/(x_bpt1-0.05)+1.3
        ax5.plot(x_bpt,y_bpt,'r')
        ax5.plot(x_bpt1,y_bpt1,'k')
        x_sf = self.x[self.mask_2]
        y_sf = self.y[self.mask_2]
        
#         t=Table([self.x[self.mask_1],self.y[self.mask_1]],names=['x','y'])
#         t.write(self.dir+'s_x_y.fits',format='fits')
        ax5.scatter(self.x[self.mask_1], self.y[self.mask_1], color='dodgerblue', s=2,label='0-1re')
        ax5.scatter(x_sf, y_sf, color='orange', s=2,label='0-1re sf')
#         ax5.axvline(x=-0.05,label='x=-0.05')
        ax5.legend()
        ax5.set_xlabel('log(NII/Ha)')
        ax5.set_ylabel('log([OIII]/Hb)')
        ax5.set_title(self.plateifu)
        ax1.plot(self.wave,self.flux_stack)
        plt.savefig(self.dir+'%s_1re_stack.jpg'%self.plateifu,format='jpg')
#         plt.show()
        plt.close()

           
    def min_snr_emission_do2(self):
        """
        DO2: NII,Ha
        KD02: OII,OIII,NII,Hb
        """       
        '''
        NII,Ha
        '''
        lines=np.array([6548.03, 6583.41,6562.80])
        dv=np.full_like(lines,800)
        c = 299792.458
        flag = False
        snr_emi=[]
        for line, dvj in zip(lines, dv):
            flag = (self.wave > line*(1 + self.zifu)*(1 - dvj/(c))) \
                & (self.wave < line*(1 + self.zifu)*(1 + dvj/(c)))
            f_emi=self.flux_stack[flag]
            err_emi=self.error_stack[flag]
            snr_emi.append(np.sum(f_emi)/np.sqrt(np.sum(err_emi)))
        if np.min(snr_emi)>5:
            with open(self.dir+'do2_min_snr_emi_large5.txt','a+') as f_do2_output:
                print(self.plateifu,file=f_do2_output)
                     
    def min_snr_emission_kd02(self):
        '''
        OII,OIII,NII,Hb
        '''
        lines=np.array([3726.03, 3728.82, 4861.33, 4958.92, 5006.84, 6548.03, 6583.41])
        dv=np.full_like(lines,800) # 800 width
        c = 299792.458
        flag = False
        snr_emi=[]
        for line, dvj in zip(lines, dv):
            flag = (self.wave > line*(1 + self.zifu)*(1 - dvj/(c))) \
                & (self.wave < line*(1 + self.zifu)*(1 + dvj/(c)))
            f_emi=self.flux_stack[flag]
            err_emi=self.error_stack[flag]
            snr_emi.append(np.sum(f_emi)/np.sqrt(np.sum(err_emi)))
        if np.min(snr_emi)>5:
            with open(self.dir+'kd02_min_snr_emi_large5.txt','a+') as f_kd02_output:
                print(self.plateifu,file=f_kd02_output)
            

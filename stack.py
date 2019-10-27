import numpy as np 
import matplotlib.pyplot as plt
import numpy as np 
from astropy.table import Table
np.seterr(divide='ignore', invalid='ignore')

class stack(object):
    """
    stacking 0-1re
    S/N>5 sigma
    """
    def __init__(self,plateifu,wave,flux,ivar,flux_map,ivar_map,mask,ellcoo,v,dirres,plot=True):
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
        
        """
        count
        """
        self.pre=self.remove_agn_badspx_ellcoo()
        f1=open(self.dir+'vaild.txt','a+')
        if self.pre>0.3:
            print(self.plateifu,file=f1)
            self.vel2z()
            self.stack()
            if plot:
                self.plot() 
        f1.close()
    def remove_agn_badspx_ellcoo(self):
        """
        remove bad spaxel in map data
        """
        mask_bad=(self.mask!=0) # !=0 remove
        self.flux_map=np.ma.array(self.flux_map,mask=mask_bad)
        self.ivar_map=np.ma.array(self.ivar_map,mask=mask_bad)
        
        """
        BPT diagram remove AGN
        Kauffmann 2003: log(OIII/hb)>0.61/(log(NII/Ha)-0.05)+1.13
        [oiii]5008,4960:13,12(5007)
        Hb4862:11
        NII6549,6585:17,19(6584)
        Ha6564:18
        """        
        x=np.log10(self.flux_map[19]/self.flux_map[18])
        y=np.log10(self.flux_map[13]/self.flux_map[11])
        mask_bpt=(y<=0.61/(x-0.05)+1.13)
        mask_bpt_valid=(y>0.61/(x-0.05)+1.13)
        #mask in agn is not same with sf
        #f=np.ma.array(f,mask=mask)
        
        """
        ellcoo
        """
        
        """
        count effective points
        """
        mask_1=(self.ellcoo<=1)
        plt.subplot(1,2,1)
#         plt.imshow(mask_1) 
        num_all=np.sum(mask_1)
        mask_2=(self.ellcoo<=1)&(mask_bpt)
        plt.imshow(mask_2)
        num_valid=np.sum(mask_2)
        mask_3=mask_1&(mask_bpt_valid)
#         plt.savefig(self.dir+'%s_1re_stack_imshow.jpg'%self.plateifu,format='jpg')
#         plt.imshow(mask_3)
        pre=num_valid/num_all
        print("percentage of vaild points : %s / %s"%(num_valid,num_all))
        print("percentage of vaild points : ",num_valid/num_all)
        """
        drp logcube
        """
        print(self.flux.shape)
        self.flux=self.flux[mask_2]
        self.v=self.v[mask_2]
        self.ivar=self.ivar[mask_2]
        
        return pre
                
    def sf(self):
        """
        Pick out SF region:
        """
        x=np.log10(self.flux_map[19]/self.flux_map[18])
        y=np.log10(self.flux_map[13]/self.flux_map[11])
        mask=(y<0.61/(x-0.05)+1.13)
        # f=f[mask]
        return mask
        
    
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
            flux=np.interp(self.wave,wave_rest,flux)
            ivar=np.interp(self.wave,wave_rest,ivar)
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
        if snr>5:
            
            print('plateifu',self.plateifu)
            table=[self.wave]
            table.append(self.flux_stack)
            table.append(self.error_stack)
            table=np.transpose(table)
            t=Table(table,names=['wave','flux','error'])
            t.write(self.dir+'%s_1re_stack.fits'%self.plateifu,format='fits')
        else:
            print('snr<5',self.plateifu)
            
    def plot(self):
        """
        plot (wave,flux)
        """
        plt.subplot(1,2,2)
        plt.plot(self.wave,self.flux_stack)
        plt.savefig(self.dir+'%s_1re_stack.jpg'%self.plateifu,format='jpg')
#         plt.show()
           
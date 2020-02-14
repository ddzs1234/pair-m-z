'''
2019-09-14 16:51:15 
还是使用mpl8_v2_3.py的code;
改动:去除最开始的拟合步骤;
加入噪声

**wave没有到rest_frame**sc
2019-09-17 16:36:37 
wave mask rest frame
wave not rest frame

'''

import glob
from os import path,mkdir
from time import perf_counter as clock

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

import ppxf as ppxf_package
import ppxf.miles_util as lib
import ppxf.ppxf_util as util
import ppxf.ppxfgas as gas
import ppxf.ppxfstellar as stellar
from ppxf.ppxf import ppxf


def ppxf_example_kinematics_sdss(dirfile, galaxy, lam_gal, plateifu, mask,
                                 noise, redshift):

    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    z = redshift
    lam_gal = lam_gal
    mask = mask

    c = 299792.458
    frac = lam_gal[1] / lam_gal[0]
    
    a = np.full((1, 4563), 2.76)
    fwhm_gal = a[0][mask]

    velscale = np.log(frac) * c

    vazdekis = glob.glob(ppxf_dir + '/miles_models/Mun1.30Z*.fits')
    fwhm_tem = 2.51
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam_temp = h2['CRVAL1'] + h2['CDELT1'] * np.arange(h2['NAXIS1'])
    lamRange_temp = [np.min(lam_temp), np.max(lam_temp)]
    sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
    templates = np.empty((sspNew.size, len(vazdekis)))
    fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)

    fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
    sigma = fwhm_dif / 2.355 / h2['CDELT1']  # Sigma difference in pixels

    for j, fname in enumerate(vazdekis):
        hdu = fits.open(fname)
        ssp = hdu[0].data
        ssp = util.gaussian_filter1d(
            ssp, sigma)  # perform convolution with variable sigma
        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates[:, j] = sspNew / np.median(sspNew)  # Normalizes templates

    c = 299792.458
    dv = np.log(lam_temp[0] / lam_gal[0]) * c  # km/s
    goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, z)

    vel = c * np.log(1 + z)  # eq.(8) of Cappellari (2017)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]

    pp = stellar.ppxf(dirfile,
                      templates,
                      galaxy,
                      noise,
                      velscale,
                      start,
                      z,
                      goodpixels=goodpixels,
                      plot=True,
                      moments=4,
                      degree=12,
                      vsyst=dv,
                      clean=False,
                      lam=lam_gal)
    return pp.bestfit, pp.lam


def emission(dirfile, w1, f1, redshift, plateifu, tie_balmer, limit_doublets):
    ppxf_dir = path.dirname(path.realpath(ppxf_package.__file__))

    z = redshift
    flux = f1
    galaxy = flux
    wave = w1

    wave *= np.median(util.vac_to_air(wave) / wave)

    noise = np.full_like(galaxy,
                         0.01635)  # Assume constant noise per pixel here

    c = 299792.458  # speed of light in km/s
    velscale = c * np.log(wave[1] / wave[0])  # eq.(8) of Cappellari (2017)
    # SDSS has an approximate instrumental resolution FWHM of 2.76A.
    FWHM_gal = 2.76

    # ------------------- Setup templates -----------------------

    pathname = ppxf_dir + '/miles_models/Mun1.30*.fits'

    # The templates are normalized to mean=1 within the FWHM of the V-band.
    # In this way the weights and mean values are light-weighted quantities
    miles = lib.miles(pathname, velscale, FWHM_gal)

    reg_dim = miles.templates.shape[1:]

    regul_err = 0.013  # Desired regularization error

    lam_range_gal = np.array([np.min(wave), np.max(wave)]) / (1 + z)
    gas_templates, gas_names, line_wave = util.emission_lines(
        miles.log_lam_temp,
        lam_range_gal,
        FWHM_gal,
        tie_balmer=tie_balmer,
        limit_doublets=limit_doublets)

    templates = gas_templates

    c = 299792.458
    dv = c * (miles.log_lam_temp[0] - np.log(wave[0])
              )  # eq.(8) of Cappellari (2017)
    vel = c * np.log(1 + z)  # eq.(8) of Cappellari (2017)
    start = [vel, 180.]  # (km/s), starting guess for [V, sigma]

    #     n_temps = stars_templates.shape[1]
    n_forbidden = np.sum(["[" in a
                          for a in gas_names])  # forbidden lines contain "[*]"
    n_balmer = len(gas_names) - n_forbidden

    component = [0] * n_balmer + [1] * n_forbidden
    gas_component = np.array(
        component) >= 0  # gas_component=True for gas templates

    moments = [2, 2]

    start = [start, start]

    gas_reddening = 0 if tie_balmer else None

    t = clock()
    pp = gas.ppxf(dirfile,
                  templates,
                  galaxy,
                  noise,
                  velscale,
                  start,
                  z,
                  plot=True,
                  moments=moments,
                  degree=-1,
                  mdegree=10,
                  vsyst=dv,
                  lam=wave,
                  clean=False,
                  component=component,
                  gas_component=gas_component,
                  gas_names=gas_names,
                  gas_reddening=gas_reddening)

    pp.plot()
    return pp.bestfit, pp.lam


def move_continuum(wave, z, width=800):
    """
    Generates a list of goodpixels to mask a given set of gas emission
    lines. This is meant to be used as input for PPXF.

    :param logLam: Natural logarithm np.log(wave) of the wavelength in
        Angstrom of each pixel of the log rebinned *galaxy* spectrum.
    :param lamRangeTemp: Two elements vectors [lamMin2, lamMax2] with the minimum
        and maximum wavelength in Angstrom in the stellar *template* used in PPXF.
    :param z: Estimate of the galaxy redshift.
    :return: vector of goodPixels to be used as input for pPXF

    """
    #                     -----[OII]-----   Hbeta   -----[OIII]-----   [OI]    -----[NII]-----   Halpha
    lines = np.array([
        3726.03, 3728.82, 4861.33, 4958.92, 5006.84, 6548.03, 6583.41, 6562.80
    ])
    dv = np.full_like(lines, width)
    c = 299792.458

    flag = False
    for line, dvj in zip(lines, dv):
        flag |= (wave > line*(1 + z)*(1 - dvj/(2*c))) \
              & (wave < line*(1 + z)*(1 + dvj/(2*c)))
    return flag


def fitting(filenames,plateifu,snr):
    
    
    dir2 = '/media/ashley/project/pair_galaxy/2020-1-31/bpt/diff_snr/'


    with fits.open('/media/ashley/project/mangawork/manga/spectro/redux/v2_5_3/drpall-v2_5_3.fits') as zfile:
        
        if not path.exists(dir2 +plateifu+'_%s/'%snr) and path.exists(filenames):
            mkdir(dir2+plateifu+'_%s/'%snr)
            dirfile = dir2 +plateifu+'_%s/'%snr+plateifu + '.txt'
            
            data = zfile[1].data
            pifu = data.field('plateifu')
            z_info = data.field('z')
            index_z = np.where(pifu == plateifu)[0]
            z1 = z_info[index_z]
        
            file = fits.open(filenames)
            t = file[1].data
            if z1.size>0:
                print(plateifu,z1)
                mask1 = (t['wave'] / (1 + z1) > 3540) & (t['wave'] /
                                                         (1 + z1) < 7409)
                
                flux = t['flux'][mask1]
                if np.median(flux) >0:
                    print(np.median(flux))
                    galaxy = flux / np.median(flux)
                    wave = t['wave'][mask1]
                    noise = t['flux_error'][mask1]
                    f1 = galaxy
                    w1 = wave
                    try:
                        f2, w2 = ppxf_example_kinematics_sdss(dirfile, galaxy, wave,
                                                              plateifu, mask1, noise,
                                                              z1)
                
                        """
                        先看看信噪比》5：
                        """
                        mask_em = move_continuum(w1, z1)
                        flux_mask = (f1 - f2)[mask_em]
                        w1_mask = w1[mask_em]
                        noise_mask = noise[mask_em]
                        snr_em = np.sum(flux_mask) / np.sqrt(np.sum(noise_mask**2))
                
                        plt.figure()
                        plt.plot(w1_mask,
                                  flux_mask,
                                  label='emission line mask %s' % snr_em)
                        plt.legend()
                        plt.savefig(dirfile[:-4] + 'snr_em.jpg', dpi=300)
                        plt.axis('off')
                        plt.clf()
                        if snr_em > 5:
                            ff, ww = emission(dirfile,
                                              w1,
                                              f1 - f2,
                                              z1,
                                              plateifu,
                                              tie_balmer=False,
                                              limit_doublets=False)
                
                            plt.figure()
                            plt.plot(ww, ff, label='fitting')
                            plt.plot(ww, galaxy - f2, ":", label='stacking')
                            plt.title(plateifu)
                            plt.legend()
                            plt.savefig(dirfile[:-4] + '_show.jpg', dpi=300)
                            plt.axis('off')
                            plt.clf()
                
                        else:
                            with open('snr<5.txt', 'a+') as f_snr:
                                print('snr<5', plateifu, file=f_snr)
                    except IndexError:
                        print('index_error',plateifu)
                        pass
            file.close()




if __name__ == '__main__':
    dirstack = '/media/ashley/project/pair_galaxy/2020-1-31/bpt/snr_stellar_velocity_fitting/'
    filenames = glob.glob(dirstack+'*.fits')
    print(filenames)
    for i in filenames:
        fitting(i)
        

    
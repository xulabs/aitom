#!/usr/bin/env python

'''
reconstruction based simulation by simply convoluting with CTF and adding proper level of noise 

code modified according to 
aitom/aitom/simulation/backprojection_reconstruction.m
aitom/aitom/simulation/reconstruction__eman2.py

import aitom.simulation.reconstruction__simple_convolution as TSRSC
aitom/aitom/simulation/reconstruction__simple_convolution.py

'''
import copy
import numpy as N
import numpy.fft as NF

import aitom.model.util as TMU
import aitom.image.vol.wedge.util as IVWU
import aitom.image.optics.ctf as TIOC
import aitom.image.vol.util as TIVU
import aitom.geometry.rotate as TGR
import aitom.geometry.ang_loc as TGAL

from aitom.simulation.reconstruction.reconstruction__util import tom_bandpass


'''
create a simulated subtomogram given density map. Add proper level of noise.


adapted from
~/ln/frequent_structure/code/GenerateSimulationMap.m
backprojection_reconstruction()
~/src/aitom/aitom/simulation/reconstruction__eman2.py
'''

def do_reconstruction(v, op, signal_variance=None, verbose=False):

    mask = IVWU.wedge_mask(v.shape, op['model']['missing_wedge_angle']) * TMU.sphere_mask(v.shape);     assert      N.all(N.isfinite(mask))

    if 'Dz' in op['ctf']:
        ctf = TIOC.create(Dz=op['ctf']['Dz'], size=v.shape, pix_size=op['ctf']['pix_size'], voltage=op['ctf']['voltage'], Cs=op['ctf']['Cs'], sigma=op['ctf']['sigma'] if 'sigma' in op['ctf'] else None)['ctf']
    else:
        # in this case, we do not have CTF defined
        ctf = N.zeros_like(mask) + 1

    if signal_variance is None:
        signal_variance = calc_variance(v_e=v, ctf=ctf, mask=mask, verbose=verbose)['variance_total']
        
    print ('signal_variance', signal_variance)

    corrfac_t = calc_corrfac(ctf=ctf, mask=mask)
    corrfac = corrfac_t['e_var'] / corrfac_t['ec_var']
    if verbose:     print ('corrfac_t', corrfac_t, 'corrfac', corrfac)

    signal_variance *= corrfac
    if verbose:     print ('signal_variance corrfac', signal_variance)

    vb = do_reconstruction_given_sigma(v=v, ctf=ctf, signal_variance=signal_variance, op=op, verbose=verbose)


    vb_t = vb.copy()
    vb_t = NF.fftn(vb_t)
    vb_t = NF.fftshift(vb_t)
    vb_t *= mask
    vb_t = NF.ifftshift(vb_t)
    vb_t = NF.ifftn(vb_t)
    vb_t = N.real(vb_t)
    assert      N.all(N.isfinite(vb_t))

    return vb_t



'''
this is to save computation, given dm which was masked by wedge mask, and  pre-calculated ctf, variance_sigma,
convolute dm with ctf with proper level of noise added
'''
def do_reconstruction_given_sigma(v, ctf, signal_variance, op, verbose=False):

    op = copy.deepcopy(op)

    Ny = 1 / (2.0 * op['ctf']['pix_size'])
    BPThresh = N.floor(((1/4.0)*(v.shape[0]/2.0))/Ny)
    if verbose: print ('BPThresh', BPThresh)
  
    v = tom_bandpass(v, low=0, hi=BPThresh, smooth=2.0)
    assert N.all(N.isfinite(v))


    WeiProj = 0.5
    WeiMTF = 1 - WeiProj


    mid_co = TIVU.fft_mid_co(v.shape)

    SNR = op['model']['SNR']
    if SNR is None:     SNR = N.inf


    if N.isfinite(SNR):
        noisy = v + N.random.normal(loc=0.0, scale=N.sqrt(WeiProj/SNR*signal_variance), size=v.shape)
    else:
        noisy = v

    noisy = NF.fftn(noisy)
    noisy = NF.fftshift(noisy)
    noisy *= ctf
    noisy = NF.ifftshift(noisy)
    noisy = NF.ifftn(noisy)
    noisy = N.real(noisy)
    assert N.all(N.isfinite(noisy))

    if N.isfinite(SNR):
        mtf_t = N.random.normal(loc=0.0, scale=N.sqrt(WeiMTF/SNR*signal_variance), size=noisy.shape)
        mtf_t = tom_bandpass(mtf_t, low=0.0, hi=1.0, smooth=0.2*noisy.shape[0])
        noisy += mtf_t
        noisy = tom_bandpass(noisy, low=0.0, hi=BPThresh, smooth=2.0)
        assert N.all(N.isfinite(noisy))


    vb = noisy.copy()

    #if ('result_standardize' in op['model']) and op['model']['result_standardize']:   vb = (vb - vb.mean()) / vb.std()

    return vb


def calc_variance(v_e, ctf, mask, verbose=False):
    re = {}

    v = NF.fftn(v_e)
    v = NF.fftshift(v)
    v *= ctf * mask         # here applying wedge mask is very important for estimating correct level of noice to be added
    v = NF.ifftshift(v)
    v = NF.ifftn(v)
    v = N.real(v)

    re['variance_total'] = v.var()

    return re


def calc_corrfac(ctf, mask):
    e = N.random.normal(loc=0.0, scale=N.sqrt(1.0), size=ctf.shape)
    e = NF.fftn(e)
    e = NF.fftshift(e)
    ec = e * ctf * mask       # error convoluted with ctf. Here applying wedge mask is very important for estimating correct level of noice to be added
    ec = NF.ifftshift(ec)
    ec = NF.ifftn(ec)
    ec = N.real(ec)

    e = NF.ifftshift(e)
    e = NF.ifftn(e)
    e = N.real(e)   # error without ctf

    return {'e_var':e.var(), 'ec_var':ec.var()}



# just a test given all parameters fixed
def simulation_test0():
    import aitom.io.file as TIF

    #op = {'model':{'missing_wedge_angle':30, 'SNR':N.nan}, 'ctf':{'pix_size':1.0, 'Dz':-15.0, 'voltage':300, 'Cs':2.2, 'sigma':0.4}}
    op = {'model':{'missing_wedge_angle':30, 'SNR':0.05}, 'ctf':{'pix_size':1.0, 'Dz':-5.0, 'voltage':300, 'Cs':2.0, 'sigma':0.4}}

    v = TMU.generate_toy_model(dim_siz=64)
    
    loc_proportion = 0.1
    loc_max = N.array(v.shape, dtype=float) * loc_proportion
    angle = TGAL.random_rotation_angle_zyz()
    loc_r = (N.random.random(3)-0.5)*loc_max

    vr = TGR.rotate(v, angle=angle, loc_r=loc_r, default_val=0.0)

    import aitom_core.simulation.reconstruction.reconstruction__simple_convolution as TSRSC
    vb = TSRSC.do_reconstruction(vr, op, verbose=True)
    print ('vb', 'mean', vb.mean(), 'std', vb.std(), 'var', vb.var())

    if True:
        vb_rep = do_reconstruction(vr, op, verbose=True)

        # calculate SNR
        import scipy.stats as SS
        vb_corr = SS.pearsonr(vb.flatten(), vb_rep.flatten())[0]
        vb_snr = 2*vb_corr / (1 - vb_corr)
        print ('SNR', 'parameter', op['model']['SNR'], 'estimated', vb_snr )         # fsc = ssnr / (2.0 + ssnr)


    #TIF.put_mrc(vb, '/tmp/vb.mrc', overwrite=True)
    #TIF.put_mrc(v, '/tmp/v.mrc', overwrite=True)

    import aitom.image.io as TIIO
    import aitom.image.vol.util as TIVU

    TIIO.save_png(TIVU.cub_img(vb)['im'], "/tmp/vb.png")
    TIIO.save_png(TIVU.cub_img(v)['im'], "/tmp/v.png")


    # save Fourier transform magnitude for inspecting wedge regions
    vb_f = NF.fftn(vb)
    vb_f = NF.fftshift(vb_f)
    vb_f = N.abs(vb_f)
    vb_f = N.log(vb_f)
    TIF.put_mrc(vb_f, '/tmp/vb-f.mrc', overwrite=True)



if __name__=='__main__':
    #test_reconstruction0()
    #test_projection()
    simulation_test0()




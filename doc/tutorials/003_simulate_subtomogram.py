"""
This script shows an example of how to
1. generate a density map containing a toy structure
2. randomly rotate and translate the map
3. convert the map to subtomogram

IMPORTANT: If you are going to use this simulated data for deep learning, please do not save the simulated data to disk.
Because simated data will consume large amount of storage. Please construct a generator to directly feed the
simulated data to your model from memory.
"""

import aitom.io.file as IF
import numpy as N
# Please Note that `aitom_core` module is in the server, not public for now
import aitom_core.simulation.reconstruction.reconstruction__simple_convolution as TSRSC
import aitom.model.util as MU
import aitom.geometry.ang_loc as GAL
import aitom.geometry.rotate as GR


# set parameters for the simulation
# op = {'model': {'missing_wedge_angle': 30, 'SNR': N.nan},
#       'ctf': {'pix_size': 1.0, 'Dz': -15.0, 'voltage': 300, 'Cs': 2.2, 'sigma': 0.4}}
op = {'model': {'missing_wedge_angle': 30, 'SNR': 0.05},
      'ctf': {'pix_size': 1.0, 'Dz': -5.0, 'voltage': 300, 'Cs': 2.0, 'sigma': 0.4}}

# generate a density map v that contains a toy structure
v = MU.generate_toy_model(dim_siz=64)  # generate a pseudo density map
print(v.shape)


# randomly rotate and translate v
loc_proportion = 0.1
loc_max = N.array(v.shape, dtype=float) * loc_proportion
angle = GAL.random_rotation_angle_zyz()
loc_r = (N.random.random(3)-0.5)*loc_max
vr = GR.rotate(v, angle=angle, loc_r=loc_r, default_val=0.0)

# generate simulated subtomogram vb from v
vb = TSRSC.do_reconstruction(vr, op, verbose=True)
print('vb', 'mean', vb.mean(), 'std', vb.std(), 'var', vb.var())

# save v and vb as 3D grey scale images
IF.put_mrc(vb, '/tmp/vb.mrc', overwrite=True)
IF.put_mrc(v, '/tmp/v.mrc', overwrite=True)

# save images of the slices of the corresponding 3D iamges for visual inspection
import aitom.image.io as IIO
import aitom.image.vol.util as IVU
IIO.save_png(IVU.cub_img(vb)['im'], "/tmp/vb.png")
IIO.save_png(IVU.cub_img(v)['im'], "/tmp/v.png")

if True:
    # verify the correctness of SNR estimation
    vb_rep = TSRSC.do_reconstruction(vr, op, verbose=True)

    import scipy.stats as SS
    # calculate SNR
    vb_corr = SS.pearsonr(vb.flatten(), vb_rep.flatten())[0]
    vb_snr = 2*vb_corr / (1 - vb_corr)
    print('SNR', 'parameter', op['model']['SNR'], 'estimated', vb_snr)  # fsc = ssnr / (2.0 + ssnr)


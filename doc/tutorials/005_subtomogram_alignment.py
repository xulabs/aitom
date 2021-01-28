'''
Alignment of two 3D grey scale images
'''

#----------------------------
# generate simulated images

import aitom.model.util as TMU
v = TMU.generate_toy_model(dim_siz=32)

import aitom.geometry.ang_loc as TGAL
import aitom.geometry.rotate as TGR
import random
import numpy as N

# randomly rotate and translate v
loc_proportion = 0.1
loc_max = N.array(v.shape, dtype=float) * loc_proportion
angle = TGAL.random_rotation_angle_zyz()
loc_r = (N.random.random(3)-0.5)*loc_max
vr = TGR.rotate(v, angle=angle, loc_r=loc_r, default_val=0.0)


#--------------------------------
# align vr against v

import aitom.align.fast.util as TAU
al = TAU.align_vols_no_mask(v, vr)
print('rigid transform of alignment', al)

vr_i = TGR.rotate(vr, angle=al['angle'], loc_r=al['loc'], default_val=0.0)   # rotate vr according to the alignment, expected to produce an image similiar to v


# save images of the slices of the corresponding 3D iamges for visual inspection
import aitom.image.io as TIIO
import aitom.image.vol.util as TIVU
TIIO.save_png(TIVU.cub_img(v)['im'], "v.png")
TIIO.save_png(TIVU.cub_img(vr)['im'], "vr.png")
TIIO.save_png(TIVU.cub_img(vr_i)['im'], "vr_i.png")


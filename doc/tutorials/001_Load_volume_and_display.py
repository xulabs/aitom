'''
Load Volume and Display image¶
change to parent directory to use aitom library

'''

import os
os.chdir("..")

'''

load mrc file using io module

example data:

http://ftp.ebi.ac.uk/pub/databases/empiar/archive/10045/data/ribosomes/AnticipatedResults/Particles/Tomograms/05/IS002_291013_005_subtomo000001.mrc

'''

import aitom.io.file as io_file
a = io_file.read_mrc_data("data/IS002_291013_005_subtomo000001.mrc")


# denoising using gaussian filter for visualization

from aitom.filter.gaussian import smooth
a = smooth(a, sigma=8)

# display image using image module

import aitom.image.vol.util as im_vol_util
a_im = im_vol_util.cub_img(a)

'''
a_im is a dict:

'im': image data
'vt': volume data

image data is type of numpy.ndarray, elements $\in$ [0, 1]
'''


print(type(a_im['im']))
print(a_im['im'].shape)
print(a_im['im'][1][1])

import matplotlib.pyplot as plt
#%matplotlib notebook

plt.imshow(a_im['im'], cmap='gray')

# save the figure into a png file
import aitom.image.io as image_io
image_io.save_png(m=a_im['im'], name='/tmp/a_im.png')

# display image using `image.util.dsp_cub`

im_vol_util.dsp_cub(a)

# save slices of a into a png file
image_io.save_cub_png(v=a, name='/tmp/a.png')


"""
Load Volume and Display image¶
change to parent directory to use aitom library
"""

import os
os.chdir("..")

# load mrc file using io module
# example data:
# http://ftp.ebi.ac.uk/pub/databases/empiar/archive/10045/data/ribosomes/AnticipatedResults/Particles/Tomograms/05/IS002_291013_005_subtomo000001.mrc
import aitom.io.file as IF
a = IF.read_mrc_data("data/IS002_291013_005_subtomo000001.mrc")


# denoising using gaussian filter for visualization
from aitom.filter.gaussian import smooth
a = smooth(a, sigma=8)

# display image using image module
import aitom.image.vol.util as IVU
'''
a_im is a dict:
    'im': image data, type of numpy.ndarray, elements in [0, 1]
    'vt': volume data
'''
a_im = IVU.cub_img(a)
print(type(a_im['im']))
print(a_im['im'].shape)
print(a_im['im'][1][1])

import matplotlib.pyplot as plt
plt.imshow(a_im['im'], cmap='gray')

# save the figure into a png file
import aitom.image.io as IIO
IIO.save_png(m=a_im['im'], name='/tmp/a_im.png')

# display image using `image.util.dsp_cub`
IVU.dsp_cub(a)

# save slices of a into a png file
IIO.save_cub_img(v=a, name='/tmp/a.png')

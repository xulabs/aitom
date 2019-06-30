'''
functions for gaussian filtering

'''

import scipy.ndimage as SN
# smoothing using scipy.ndimage.gaussian_filter
def smooth(v, sigma):
    assert  sigma > 0
    return SN.gaussian_filter(input=v, sigma=sigma)


'''
Difference of gaussian filter
'''
def dog_smooth(v, s1, s2=None):
    if s2 is None:      s2 = s1 * 1.1       # the 1.1 is according to a DoG particle picking paper
    assert      s1 < s2
    return  smooth(v, s1) - smooth(v, s2)

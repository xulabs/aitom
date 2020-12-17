import numpy as np
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology


def local_maxima(arr):
    # http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local maximum filter; all locations of maximal value in their neighborhood are set to 1
    # local_max is a mask that contains the peaks we are looking for, but also the background. In order to isolate
    # the peaks we must remove the background from the mask.
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#maximum_filter
    local_max = (filters.maximum_filter(arr, footprint=neighborhood) == arr)

    # we create the mask of the background
    # mxu: in the original version, was background = (arr==0)
    background = (arr == arr.min())

    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(background, structure=neighborhood, border_value=1)

    # # we obtain the final mask, containing only peaks, by removing the background from the local_max mask
    # # mxu: this is the old version, but the boolean minus operator is deprecated
    # detected_maxima = local_max - eroded_backround

    # Material nonimplication, see http://en.wikipedia.org/wiki/Material_nonimplication
    detected_maxima = np.bitwise_and(local_max, np.bitwise_not(eroded_background))
    return np.where(detected_maxima)


def local_minima(arr):
    return local_maxima(-arr)

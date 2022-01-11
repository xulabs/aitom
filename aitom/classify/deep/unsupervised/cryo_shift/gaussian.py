import numpy as np
def gaussian3D(sigma, start, stop, n):
    xaxis = np.linspace(start, stop, n)
    yaxis = np.linspace(start, stop, n)
    zaxis = np.linspace(start, stop, n)
    x = xaxis[:,None,None]
    y = yaxis[None,:,None]
    z = zaxis[None,None,:]

    sigma_square = sigma**2

    return np.exp(-(x**2 + y**2 + z**2)/(2.0*sigma_square))/(np.sqrt(2.0*np.pi*sigma_square))

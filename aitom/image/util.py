"""
utility functions for images
"""


# display an image
def dsp_img(v, new_figure=True):
    import matplotlib.pyplot as plt

    if new_figure:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = plt

    import matplotlib.cm as cm

    ax_u = ax.imshow(v, cmap=cm.Greys_r)
    ax.axis('off')  # clear x- and y-axes

    # calling pause will display the figure without blocking the program,
    # see segmentation.active_contour.morphsnakes.evolve_visual
    plt.pause(0.001)

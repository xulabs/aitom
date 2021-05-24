'''
Author: Guanan Zhao
'''


from mayavi import mlab
import numpy as np
import cv2
from scipy import ndimage

# Generate an image for 3D data. The upper half of the image is the
#  parallel projection of contours, and the lower half shows the
#  corresponding 2D slices.
def plot_data_3d(voxel, savepath, show_image=False):
    # Draw the lower half
    slides = plot_slides(voxel)

    # Draw the upper half
    fig = mlab.figure( size=(slides.shape[1], slides.shape[1]) )
    fig.scene.parallel_projection = True
    ## Being a little tricky, we plot a totally transparent box for the 3D image
    ##  otherwise the camera would move and zoom to fit, which is not what we want
    a, b, c = voxel.shape
    virtual_box = np.zeros((a, b, c))
    virtual_box[1:a-1, 1:b-1, 1:c-1] = np.ones((a-2, b-2, c-2))
    mlab.contour3d(virtual_box, contours=[0.5], opacity=0)
    ## We then plot the data isosurface
    mlab.contour3d(voxel, color=(1,1,1), contours=[np.min(voxel)*0.3], opacity=0.8)
    ## Adjust the scene camera
    mlab.view(270, 0)
    fig.scene.camera.zoom(1.5)
    mlab.savefig(savepath)
    
    # Concatenate the two halves and save the figure
    scr = cv2.imread(savepath)
    figure = np.concatenate( (scr, slides), axis=0 )
    cv2.imwrite(savepath, figure)
    if show_image:
        mlab.show()
    else:
        mlab.close(fig)
    return figure


# Generate an image for 3D data overlapped with the CAM heatmap. 
#  The upper half of the image is the parallel projection of contours, 
#  and the lower half shows the corresponding 2D slices.
def plot_data_cam_3d(voxel, cam, savepath, show_image=False):
    # Resize the CAM
    cam_zoom = ndimage.zoom(cam, zoom = [ float(x)/y for x, y in zip(voxel.shape, cam.shape) ])

    # Draw the lower half
    slides = plot_slides(cam_zoom, colored=True)

    # Draw the upper half
    fig = mlab.figure( size=(slides.shape[1], slides.shape[1]) )
    fig.scene.parallel_projection = True
    ## Being a little tricky, we plot a totally transparent box for the 3D image
    ##  otherwise the camera would move and zoom to fit, which is not what we want
    a, b, c = voxel.shape
    virtual_box = np.zeros((a, b, c))
    virtual_box[1:a-1, 1:b-1, 1:c-1] = np.ones((a-2, b-2, c-2))
    mlab.contour3d(virtual_box, contours=[0.5], opacity=0)
    ## We then plot the data isosurface
    mlab.contour3d(voxel, color=(1,1,1), contours=[np.min(voxel)*0.3], opacity=0.5)
    ## And then the CAM heatmap (colorful=positive, black=negative)
    scale = np.max(np.abs(cam_zoom)) + 1e-10
    mlab.contour3d(np.maximum(-cam_zoom,0) / scale, color=(0,0,0), contours=[0.1, 0.3, 0.5, 0.7, 0.9], opacity=0.3)
    mlab.contour3d(np.maximum(cam_zoom,0) / scale, contours=[0.1, 0.3, 0.5, 0.7, 0.9], opacity=0.3)
    ## Adjust the scene camera
    mlab.view(270, 0)
    fig.scene.camera.zoom(1.5)
    mlab.savefig(savepath)

    # Concatenate the two halves and save the figure
    scr = cv2.imread(savepath)
    figure = np.concatenate( (scr, slides), axis=0 )
    cv2.imwrite(savepath, figure)
    if show_image:
        mlab.show()
    else:
        mlab.close(fig)
    return figure


# Plot the 2D slides of 3D data
def plot_slides(v, _range=None, colored=False):
    # Rescale the value of voxels into [0, 255], as unsigned byte
    if _range == None:
        v_n = v / np.max(np.abs(v))
        v_n = (128 + v_n * 127).astype(int)
    else:
        v_n = (v - _range[0]) / (_range[1] - _range[0])
        v_n = (v_n * 255).astype(int)

    # Plot the slides
    h, w, d = v.shape
    side_w = int(np.ceil(np.sqrt(d)))
    side_h = int(np.ceil(float(d) / side_w))
    
    board = np.zeros(( (h+1)*side_h, (w+1)*side_w, 3 ))
    if colored: # we mix jet colormap for positive part, and use pure grey-scale for negative part
        for i in range(side_h):
            for j in range(side_w):
                if i*side_w+j >= d:
                    break
                values = v_n[:,:,i*side_w+j]
                block1 = cv2.applyColorMap( np.uint8( np.maximum(0, values - 128) * 2 ), cv2.COLORMAP_JET )
                block2 = np.minimum(128, values)[:,:,np.newaxis] * np.ones( (1,1,3) )
                block = ( block1 * np.maximum(0, values-128)[:,:,np.newaxis] / 128. \
                    + block2 * np.minimum(128, 256-values)[:,:,np.newaxis] / 128. ).astype(int)
                board[ (h+1)*i+1 : (h+1)*(i+1), (w+1)*j+1 : (w+1)*(j+1), : ] = block
    else: # we just use pure grey-scale for all pixels
        for i in range(side_h):
            for j in range(side_w):
                if i*side_w+j >= d:
                    break
                for k in range(3):
                    board[ (h+1)*i+1 : (h+1)*(i+1), (w+1)*j+1 : (w+1)*(j+1), k ] = v_n[:,:,i*side_w+j]

    # Return a 2D array representing the image pixels
    return board.astype(int)


# Overlap the CAM heatmap on a 2D image
def plot_data_cam_2d(image, cam, savepath):
    cam_zoom = cv2.resize(cam, (image.shape[1], image.shape[0]))
    ## Here for natural images, we only focus on positive values of CAM
    cam_n = np.maximum(cam_zoom, 0) / np.max(cam_zoom)

    heatmap = cv2.applyColorMap(np.uint8(255*cam_n), cv2.COLORMAP_JET)
    overlapped = np.float32(heatmap) + np.float32(image)
    figure = np.uint8(255 * overlapped / np.max(overlapped))
    cv2.imwrite(savepath, figure)
    return figure

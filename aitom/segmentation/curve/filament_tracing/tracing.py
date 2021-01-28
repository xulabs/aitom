"""
Implemented based on:
1. Rigort, A., Günther, D., Hegerl, R., Baum, D., Weber, B., Prohaska, S., Medalia, O., Baumeister, W., & Hege,
H. C. (2012).

2. Automated segmentation of electron tomograms for a quantitative description of actin filament networks. Journal of
structural biology, 177(1), 135–144. https://doi.org/10.1016/j.jsb.2011.08.012

Test data from:
1. Vinzenz, M., Nemethova, M., Schur, F., Mueller, J., Narita, A., Urban, E., Winkler, C., Schmeiser, C., Koestler,
S. A., Rottner, K., Resch, G. P., Maeda, Y., & Small, J. V. (2012).

2. Actin branching in the initiation and maintenance of lamellipodia. Journal of cell science, 125(Pt 11),
2775–2785. https://doi.org/10.1242/jcs.107623

The search cone can be found at: https://cmu.box.com/s/eg0tr9m1jkar1wsgjmkajgqqbcytrlt2
"""

import numpy as np
import aitom.geometry.rotate as GR
import aitom.geometry.ang_loc as AA
import math
import mrcfile
import sys


def angle_between(u, v):
    """
    calculates the angle between two vectors
    @param:
        u: vector as a np.array
        v: vector as a np.array

    @return:
        angle between u and v in radian
    """
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    angle = np.arccos(np.clip(cos, -1, 1))
    return angle


def similarity(x, x0, corr, sigma_c, sigma_l, sigma_d, orientation):
    """
    Similarity function, i.e. "the likelihood for a voxel x to be part of the same filament" as x0
    @param:
        x: index of voxel x
        x0: index of voxel x0
        corr: correlation coefficient at voxel x from template matching
        sigma_c: user-defined parameter for smoothness (smoother lines with smaller sigma_c)
        sigma_l: user-defined parameter for linearity
        sigma_d: user-defined parameter for distance
        orientation: a numpy array storing the template orientations in vector form from template matching

    @return:
        a similarity value that measures the likelihood for x0 and x to be part of the same filament
    """
    # first get vector x-x0
    # vec = x - x0 but as a np.array
    vec = np.array([x[0] - x0[0], x[1] - x0[1], x[2] - x0[2]])
    beta = angle_between(vec, orientation[x0])
    gamma = angle_between(vec, orientation[x])

    # Gaussian functions
    # co-circularity: smootheness
    C = math.exp((-(beta - gamma) ** 2) / (sigma_c ** 2))
    # linearity
    L = math.exp((-(beta + gamma) ** 2) / (sigma_l ** 2))
    # distance
    D = math.exp((-(np.linalg.norm(vec) ** 2)) / (sigma_d ** 2))

    # similarity function
    s = corr * C * L * D
    return s


def convert(index):
    """
    converts index in the form of a numpy array to a tuple
    e.g. (array([0]),array([0]),array([0]))-->(0,0,0)

    @param:
        index: the index to be converted

    @return:
        a index tuple
    """
    return index[0][0], index[1][0], index[2][0]


def forward(x0, sc, c, phi, theta, psi, orientation, sigma_c, sigma_l, sigma_d, t1, bm):
    """
    The forward tracing algorithm (recursive)
    @param:
        x0: index of the starting voxel
        sc: the forward (along y) search cone as a numpy array
        c: a numpy array storing the correlation coefficients from template matching
        phi: a numpy array storing the first ZYZ rotation angle of the template from template matching
        theta: a numpy array storing the second ZYZ rotation angle of the template from template matching
        psi: a numpy array storing the thrid ZYZ rotation angle of the template from template matching
        orientaion: a numpy array storing the template orientations in vector form from template matching
        sigma_c: user-defined parameter for smoothness (smoother lines with smaller sigma_c)
        sigma_l:	user-defined parameter for linearity
        sigma_d: user-defined parameter for distance
        t1: threshold for similarity
        bm: a numpy array storing the binary mask that is to be updated

    @returns:
        bm when search_result.max() <= t1
    """
    # get ZYZ template rotation angles
    angles = (phi[x0], theta[x0], psi[x0])
    # rotate search cone according to template rotation angles
    sc_curr = GR.rotate(sc, angle=angles, default_val=0.0)
    # get x0 location on the search cone for index calculation later
    x0_loc = convert(np.where(sc_curr == sc_curr.min()))
    # convert the format of sc: >0-->True
    sc_mask = sc_curr > 0
    # create a np array the size of the sc to store search results
    # highest similarity in search_result means this voxel is part of the filament
    search_result = np.full(sc_mask.shape, -1.0)

    # for every voxel in the search cone, calculate similarity and update search_result
    for idx, val in np.ndenumerate(sc_mask):
        if val is True:
            # x = idx - x0_loc + x0 #convert to the index of x
            x = (x0[0] - x0_loc[0] + idx[0], x0[1] - x0_loc[1] + idx[1], x0[2] - x0_loc[2] + idx[2])
            # z, y, x boundaries
            if (x[0] <= c.shape[0] - 1) and (x[1] <= c.shape[1] - 1) and (x[2] <= c.shape[2] - 1):
                # calculate similarity between x and x0
                s = similarity(x0, x, c[x], sigma_c, sigma_l, sigma_d, orientation)
                if s > t1:
                    # store voxel location when similarity is above threshold t1
                    search_result[idx] = s

    # continue forward()
    if search_result.max() > t1:
        # highest similarity in the search cone
        x = convert(np.where(search_result == search_result.max()))
        # convert coordinate
        x = (x0[0] - x0_loc[0] + x[0], x0[1] - x0_loc[1] + x[1], x0[2] - x0_loc[2] + x[2])
        # update binary mask
        bm[x] = True
        # recursively move
        return forward(x, sc, c, phi, theta, psi, orientation, sigma_c, sigma_l, sigma_d, t1, bm)
    # forward
    # stop searching when all similarity values within sc are below threshold t1
    else:
        return bm


def backward(x0, sc, c, phi, theta, psi, orientation, sigma_c, sigma_l, sigma_d, t1, bm):
    """
    The backward tracing algorithm (recursive)
    @param:
        x0: index of the starting voxel
        sc: the backward (along -y) search cone as a numpy array
        c: a numpy array storing the correlation coefficients from template matching
        phi: a numpy array storing the first ZYZ rotation angle of the template from template matching
        theta: a numpy array storing the second ZYZ rotation angle of the template from template matching
        psi: a numpy array storing the thrid ZYZ rotation angle of the template from template matching
        orientation: a numpy array storing the template orientations in vector form from template matching
        sigma_c: user-defined parameter for smoothness (smoother lines with smaller sigma_c)
        sigma_l:	user-defined parameter for linearity
        sigma_d: user-defined parameter for distance
        t1: threshold for similarity
        bm: a numpy array storing the binary mask that is to be updated

    @returns:
        bm when search_result.max() <= t1
    """
    angles = (phi[x0], theta[x0], psi[x0])
    # rotate search cone according to template orientation
    # search cone rotation
    sc_curr = GR.rotate(sc, angle=angles, default_val=0.0)
    # get x0 location on the search cone for index calculation
    x0_loc = convert(np.where(sc_curr == sc_curr.min()))
    # later
    # convert the format of sc: >0-->True
    sc_mask = sc_curr > 0
    # create a np array the size of the sc to store search results
    # highest similarity in search_result means this voxel is part of the filament
    search_result = np.full(sc_mask.shape, -1.0)

    # for every voxel in the search cone, calculate similarity and update search_result
    for idx, val in np.ndenumerate(sc_mask):
        if val is True:
            # x = idx - x0_loc + x0 #convert to the index of x
            x = (x0[0] - x0_loc[0] + idx[0], x0[1] - x0_loc[1] + idx[1], x0[2] - x0_loc[2] + idx[2])
            # z, y, x boundaries
            if (x[0] <= c.shape[0] - 1) and (x[1] <= c.shape[1] - 1) and (x[2] <= c.shape[2] - 1):
                s = similarity(x0, x, c[x], sigma_c, sigma_l, sigma_d, orientation)
                if s > t1:
                    # store voxel location when similarity is above threshold t1
                    search_result[idx] = s

    # continue backward()
    if search_result.max() > t1:
        # highest similarity in the search cone
        x = convert(np.where(search_result == search_result.max()))
        # convert coordinate
        x = (x0[0] - x0_loc[0] + x[0], x0[1] - x0_loc[1] + x[1], x0[2] - x0_loc[2] + x[2])
        # update binary mask
        bm[x] = True
        # recursively move
        return backward(x, sc, c, phi, theta, psi, orientation, sigma_c, sigma_l, sigma_d, t1, bm)
    # backward
    # stop searching when all similarity values within sc are below threshold t1
    else:
        return bm


def trace(t1, t2, sigma_c, sigma_l, sigma_d, sc_path, out_path, c_path, phi_path, theta_path, psi_path):
    """
    The main function for filament tracing
    @param:
        t1: threshold for similarity, defualt=0.000001
        t2: threshold for correlation coefficient, default= 0.0007
        sigma_c: measure for smootheness, default=1.0
        sigma_l: measure for linearity, default=1.0
        sigma_d: measure for distance, default=1.0
        sc_path: file path of search cone, see search cone linked at the top of the script
        out_path: output file path for the tracing result (a binary mask)
        c_path: file path of the correlation coefficients (e.g. c.npy from 007_template_matching.py)
        phi_path: file path of the first ZYZ rotation angle(e.g. phi.npy from 007_template_matching.py)
        theta_path: file path of the second ZYZ rotation angle(e.g. theta.npy from 007_template_matching.py)
        psi_path: file path of the thrid ZYZ rotation angle(e.g. psi.npy from 007_template_matching.py)

    @return:
        bm, a binary mask storing the search result

    Note that if out_path is not None, trace() would store the search result at the given location
    """
    # forward() and backward() might reach recursion limit
    sys.setrecursionlimit(10000)

    assert (sc_path is not None) and (c_path is not None), "File path error"
    assert (phi_path is not None) and (theta_path is not None) and (psi_path is not None), "File path error"
    if t1 is None:
        t1 = 0.000001
    if t2 is None:
        t2 = 0.0007
    if sigma_c is None:
        sigma_c = 1.0
    if sigma_l is None:
        sigma_l = 1.0
    if sigma_d is None:
        sigma_d = 1.0

    # load template matching results (see 007_template_matching_tutorial.py)
    print("Loading template matching results...")
    c = np.load(c_path)  # (10, 400, 398)
    phi = np.load(phi_path)
    theta = np.load(theta_path)
    psi = np.load(psi_path)
    print("Map shape: ", c.shape)

    # preprocess cross correlation matrix: negative values => 0.0
    print("Preproces correlation coefficients...")
    c[c < 0.0] = 0.0
    print("Maximum corr = ", c.max())

    print("Preprocessing orientations...")
    # orientation = np.load('./orietnation.npy', allow_pickle=True)#can store orientation.npy for faster testing
    # empty np array to store template orientations
    orientation = np.empty(c.shape, dtype=object)
    # assume template is parallel to the y axis
    v = np.array([0, 1, 0])
    # for all voxels
    for idx, x in np.ndenumerate(c):
        # get ZYZ angle for template orientation
        angle = (phi[idx], theta[idx], psi[idx])
        # convert to rotation matrix
        rm = AA.rotation_matrix_zyz(angle)
        # template orientation in [x, y, z]
        orientation[idx] = rm.dot(v)

        # binary mask to store tracing results
    bm = np.full(c.shape, False, dtype=bool)

    # read in search cone: SC is along the y axis, i.e. [0,1,0]
    print("Preparing search cone...")
    mrc = mrcfile.open(sc_path, mode='r+', permissive=True)
    # rotated already, along the y axis for now #forward search cone
    sc = mrc.data
    print('search cone size', sc.shape)  # (10, 11, 11)
    # mark where x0 is with a negative value
    sc[(5, 0, 5)] = -100.0

    # 180 degree rotation of the search cone
    rm = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    # about the XYZ axes
    # rotate to get the backward search cone
    sc_back = GR.rotate(sc, rm=rm, default_val=0.0)

    # tracing: go through corr for all c.max()>=t2
    print("Start tracing actin filaments...")
    while c.max() >= t2:
        # tracing starts with the highest corr
        corr = c.max()
        # get the index of c.max #e.g.(array([5]), array([177]), array([182]))
        x0 = np.where(c == corr)
        assert (x0[0].size == 1), "multiple maximum"
        # convert index to a tuple
        x0 = (x0[0][0], x0[1][0], x0[2][0])
        print("Now at ", x0, " with corr=", corr)
        bm[x0] = True

        # recursively search forward
        bm = forward(x0, sc, c, phi, theta, psi, orientation, sigma_c, sigma_l, sigma_d, t1, bm)
        bm = backward(x0, sc_back, c, phi, theta, psi, orientation, sigma_c, sigma_l, sigma_d, t1, bm)
        # set c.max() to -1.0 and go to the next c.max()
        c[x0] = -1.0

    # output bm
    if out_path is not None:
        print("Generating output to ", out_path)
        bm = bm.astype(np.int16)
        mrc = mrcfile.new(out_path, overwrite=True)
        mrc.set_data(bm)
        mrc.close()
    print("TRACING DONE")
    return bm

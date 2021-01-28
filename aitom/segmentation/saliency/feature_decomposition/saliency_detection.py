import scipy.ndimage as SN
import numpy as np
import heapq
import aitom.io.file as io_file
import matplotlib.pyplot as plt
import time
import math
import numpy.linalg
import multiprocessing
import gc
from numba import jit
from scipy.spatial.distance import cdist
from scipy import signal
import mrcfile


def saliency_detection(a, gaussian_sigma, gabor_sigma, gabor_lambda, cluster_center_number,
                       pick_num=None, multiprocessing_num=0, save_flag=False):
    """
    saliency detection

    @param:
        a: volume data
        gaussian_sigma: sigma for de-noise
        gabor_sigma/gabor_lambda: sigma/lambda for Gabor filter
        cluster_center_number: initial number of cluster centers
        ave_flag: set True to save results
        pick_num: the number of particles to pick out

    @return:
        saliency map, the same shape as a
    """
    # Step 1 Data Pre-processing
    b_time = time.time()
    # de-noise
    a = SN.gaussian_filter(input=a, sigma=gaussian_sigma)
    end_time = time.time()
    print('de-noise takes', end_time - b_time, 's', ' sigma=', gaussian_sigma)
    original_tom = a
    if save_flag:
        img = (a[:, :, int(a.shape[2] / 2)]).copy()
        plt.imsave('/tmp/result/original.png', img, cmap='gray')

    # Step 2 Supervoxel over-segmentation
    N = a.shape[0] * a.shape[1] * a.shape[2]
    n = cluster_center_number
    # cluster center [x y z g]
    ck = []
    interval = int(math.pow(N / n, 1.0 / 3))
    x = int(interval / 2)
    y = int(interval / 2)
    z = int(interval / 2)
    print('interval=%d' % interval)

    # Initialization
    while (x < a.shape[0]) and (y < a.shape[1]) and (z < a.shape[2]):
        ck.append([x, y, z, a[x][y][z]])
        if x + interval < a.shape[0]:
            x = x + interval
        elif y + interval < a.shape[1]:
            x = int(interval / 2)
            y = y + interval
        else:
            x = int(interval / 2)
            y = int(interval / 2)
            z = z + interval

    print('the number of cluster centers = %d' % len(ck))
    print(ck[: 5])
    label = np.full((a.shape[0], a.shape[1], a.shape[2]), 0)
    distance = np.full((a.shape[0], a.shape[1], a.shape[2]), np.inf)
    start_time = time.time()
    print('Supervoxel over-segmentation begins')
    ck = np.array(ck)
    redundant_flag = np.array([False] * len(ck))

    # 10 iterations suffices for most images
    for number in range(10):
        b_time = time.time()
        print('\n%d of 10 iterations' % number)
        distance, label, ck, redundant_flag = fast_SLIC(distance, label, ck, a, interval, redundant_flag)
        # merge cluster centers
        # merge two cluster centers if the distance between them is less than ck_dist_min
        ck_dist_min = interval / 2
        for ck_i in range(len(ck)):
            if redundant_flag[ck_i]:
                continue
            d = cdist(ck[:, :3], np.reshape(ck[ck_i, :3], (1, -1))).flatten()
            ind = np.where(d < ck_dist_min)[0]
            if ind.size > 1:
                for ind_t in ind:
                    if ind_t == ck_i:
                        continue
                    redundant_flag[ind_t] = True
                    label[label == ind_t] = ck_i
                    ck[ind_t][3] = np.inf
        print('total number of remove cluster centers = ', sum(redundant_flag == True))
        e_time = time.time()
        print('\n', e_time - b_time, 's')

    cluster_center_number = int(len(ck) - sum(redundant_flag))
    # renumber cluster center index
    label = renumber(redundant_flag=redundant_flag, label=label)
    assert np.max(label) == cluster_center_number - 1
    end_time = time.time()
    print('Supervoxel over-segmentation done,', end_time - start_time, 's')

    if save_flag:
        np.save('Desktop/result/label', label)
        img = (a[:, :, int(a.shape[2] / 2)]).copy()
        k = int(a.shape[2] / 2)
        draw_color = np.min(a)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if label[i][j][k] != label[i - 1][j][k] or label[i][j][k] != label[i + 1][j][k] or label[i][j][k] != \
                        label[i][j - 1][k] or label[i][j][k] != label[i][j + 1][k]:
                    img[i][j] = draw_color
        plt.imsave('Desktop/result/SLIC.png', img, cmap='gray')
    del distance
    gc.collect()

    # Step 3
    # Feature Extraction
    stime = time.time()
    Lambda = gabor_lambda
    filters = filter_bank_gb3d(sigma=gabor_sigma, Lambda=Lambda, psi=0, gamma=1)
    # Note: For better performance, two filters with different sigmas are used here in the paper.
    # filters1 = filter_bank_gb3d(sigma=s1, Lambda=Lambda,psi=0,gamma=1)
    # filters2 = filter_bank_gb3d(sigma=s2, Lambda=Lambda,psi=0,gamma=1)
    # filters = filters1 + filters2
    filters_num = len(filters)
    # Gabor features and 6 density features
    feature_matrix = np.zeros((len(filters) + 6, cluster_center_number))
    print('%d Gabor based features' % filters_num)

    print('Feature extraction begins')
    # 3D Gabor filter based features
    res_pool = []
    label = np.load('Desktop/result/label.npy')
    if multiprocessing_num > 1:
        pool = multiprocessing.Pool(processes=min(multiprocessing_num, multiprocessing.cpu_count()))
    else:
        pool = None

    if pool is not None:
        for fm_i in range(len(filters)):
            res_pool.append(pool.apply_async(func=gabor_feature_single_job,
                                             kwds={'a': a, 'filters': filters, 'fm_i': fm_i, 'label': label,
                                                   'cluster_center_number': cluster_center_number, 'save_flag': False}))
            # res_pool.append(pool.apply_async(func=gabor_feature_single_job,
            # kwds={'a': a, 'filters': filters, 'fm_i': fm_i, 'label': np.load('Desktop/result/label.npy'),
            # 'cluster_center_number': cluster_center_number, 'save_flag': False}))
        pool.close()
        pool.join()
        del pool
        for pool_i in res_pool:
            feature_matrix[pool_i.get()[0], :] = pool_i.get()[1]
    else:
        for fm_i in range(len(filters)):
            _, feature_matrix[fm_i, :] = gabor_feature_single_job(a=a, filters=filters, fm_i=fm_i, label=label,
                                                                  cluster_center_number=cluster_center_number,
                                                                  save_flag=False)
    print('3D Gabor filter based features done')

    # density features
    feature_matrix = density_feature(a=a, feature_matrix=feature_matrix, label=label, filters_num=filters_num)
    print('Density features done')

    if save_flag:
        np.save('Desktop/result/feature_matrix', feature_matrix)

    etime = time.time()
    print('Feature extraction done,', etime - stime, 's')

    # Step 4 RPCA
    start_time = time.time()
    print('RPCA begins')
    L, S = robust_pca(feature_matrix)
    end_time = time.time()
    print('RPCA done, ', end_time - start_time, 's')
    supervoxel_saliency = np.sum(S, axis=0) / S.shape[0]
    if save_flag:
        np.save('Desktop/result/supervoxel_saliency', supervoxel_saliency)

    # Step 5 Generate Saliency Map
    saliency_map = generate_saliency_map(a=a, label=label, supervoxel_saliency=supervoxel_saliency, pick_num=pick_num)
    if save_flag:
        img = a[:, :, int(a.shape[2] / 2)].copy()
        plt.imsave('Desktop/result/saliency_map.png', img, cmap='gray')
        # io_file.put_mrc_data(a, './saliency_map.mrc')
        print('saliency map saved')

    # return saliency_map

    # Step 6 Save subtomograms
    max_saliency = np.max(supervoxel_saliency)
    min_saliency = np.min(supervoxel_saliency)
    particle_picking(a=original_tom, saliency_map=saliency_map, ref_saliency_max=max_saliency,
                     ref_saliency_min=min_saliency)
    print('subtomograms saved')


def gabor_fn(sigma, theta, Lambda, psi, gamma, size):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma
    sigma_z = float(sigma) / gamma

    # Bounding box
    (z, y, x) = np.meshgrid(np.arange(-size, size + 1), np.arange(-size, size + 1), np.arange(-size, size + 1))

    # Rotation
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    z_prime = z * R[0, 0] + y * R[0, 1] + x * R[0, 2]
    y_prime = z * R[1, 0] + y * R[1, 1] + x * R[1, 2]
    x_prime = z * R[2, 0] + y * R[2, 1] + x * R[2, 2]

    gb = np.exp(-.5 * (x_prime ** 2 / sigma_x ** 2 + y_prime ** 2 / sigma_y ** 2 + z_prime ** 2 / sigma_z)) * np.cos(
        2 * np.pi * x_prime / Lambda + psi)

    return gb


def filter_bank_gb3d(sigma, Lambda, psi=0, gamma=1, truncate=4.0):
    filters = []
    size = int(truncate * sigma + 0.5)
    '''
    for theta_x in np.arange(0, np.pi, np.pi / 4):
        for theta_y in np.arange(0, np.pi, np.pi / 4):
            for theta_z in np.arange(0, np.pi, np.pi / 4):
                thetas = [theta_x, theta_y, theta_z]
                kern = gabor_fn(sigma, thetas, Lambda, psi, gamma, size)
                kern /= kern.sum()
                filters.append(np.transpose(kern))
    '''
    # x axis
    for theta in np.arange(0, np.pi, np.pi / 4):
        thetas = [theta, 0, 0]
        # print(thetas)
        kern = gabor_fn(sigma, thetas, Lambda, psi, gamma, size)
        kern /= kern.sum()
        # kern /= 1.5 * kern.sum()
        filters.append(np.transpose(kern))
    # y axis
    for theta in np.arange(0, np.pi, np.pi / 4):
        thetas = [0, theta, np.pi / 2]
        # print(thetas)
        kern = gabor_fn(sigma, thetas, Lambda, psi, gamma, size)
        kern /= kern.sum()
        # kern /= 1.5 * kern.sum()
        filters.append(np.transpose(kern))
    # z axis
    for theta in np.arange(0, np.pi, np.pi / 4):
        thetas = [0, np.pi / 2, theta]
        # print(thetas)
        kern = gabor_fn(sigma, thetas, Lambda, psi, gamma, size)
        kern /= kern.sum()
        # kern /= 1.5 * kern.sum()
        filters.append(np.transpose(kern))

    return filters


@jit(nopython=True)
def robust_pca(M):
    """
    Decompose a matrix into low rank and sparse components.
    Computes the RPCA decomposition using Alternating Lagrangian Multipliers.
    Returns L,S the low rank and sparse components respectively
    """
    L = numpy.zeros(M.shape)
    S = numpy.zeros(M.shape)
    Y = numpy.zeros(M.shape)
    print(M.shape)
    mu = (M.shape[0] * M.shape[1]) / (4.0 * L1Norm(M))
    lamb = max(M.shape) ** -0.5
    number_of_iterations = 0
    L = svd_shrink(M - S - (mu ** -1) * Y, mu)
    S = shrink(M - L + (mu ** -1) * Y, lamb * mu)
    Y = Y + mu * (M - L - S)
    initial_error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    while not converged(M, L, S, initial_error):
        L = svd_shrink(M - S - (mu ** -1) * Y, mu)
        S = shrink(M - L + (mu ** -1) * Y, lamb * mu)
        Y = Y + mu * (M - L - S)
        number_of_iterations += 1
        if number_of_iterations >= 10000:
            break
    print(number_of_iterations, 'iterations of robust pca')
    return L, S


@jit(nopython=True)
def svd_shrink(X, tau):
    """
    Apply the shrinkage operator to the singular values obtained from the SVD of X.
    The parameter tau is used as the scaling parameter to the shrink function.
    Returns the matrix obtained by computing U * shrink(s) * V where
        U are the left singular vectors of X
        V are the right singular vectors of X
        s are the singular values as a diagonal matrix
    """
    U, s, V = numpy.linalg.svd(X, full_matrices=False)
    return numpy.dot(U, numpy.dot(numpy.diag(shrink(s, tau)), V))


@jit(nopython=True)
def shrink(X, tau):
    """
    Apply the shrinkage operator the the elements of X.
    Returns V such that V[i,j] = max(abs(X[i,j]) - tau,0).
    """
    V = numpy.copy(X).reshape(X.size)
    for i in range(V.size):
        V[i] = math.copysign(max(abs(V[i]) - tau, 0), V[i])
        if V[i] == -0:
            V[i] = 0
    return V.reshape(X.shape)


@jit(nopython=True)
def frobeniusNorm(X):
    """
    Evaluate the Frobenius norm of X
    Returns sqrt(sum_i sum_j X[i,j] ^ 2)
    """
    accum = 0
    V = numpy.reshape(X, X.size)
    for i in range(V.size):
        accum += abs(V[i] ** 2)
    return math.sqrt(accum)


@jit(nopython=True)
def L1Norm(X):
    """
    Evaluate the L1 norm of X
    Returns the max over the sum of each column of X
    """
    return max(numpy.sum(X, axis=0))


@jit(nopython=True)
def converged(M, L, S, initial_error):
    """
     A simple test of convergence based on accuracy of matrix reconstruction
    from sparse and low rank parts
    In practice, a fixed error may cause problems
    """
    error = frobeniusNorm(M - L - S) / frobeniusNorm(M)
    # print("error =", error)
    return error <= initial_error * 10e-4


@jit(nopython=True)
def fast_SLIC(distance, label, ck, a, interval, redundant_flag):
    m = 10  # m can be in the range [1,40]
    for i in range(len(ck)):
        boundary = []
        for L in range(3):
            boundary.append(int(max(ck[i][L] - interval, 0)))
            boundary.append(int(min(ck[i][L] + interval, a.shape[L])))

        for ix in range(boundary[0], boundary[1]):
            for iy in range(boundary[2], boundary[3]):
                for iz in range(boundary[4], boundary[5]):
                    dc2 = (int(a[ix][iy][iz]) - ck[i][3]) ** 2
                    ds2 = (ck[i][0] - ix) ** 2 + (ck[i][1] - iy) ** 2 + (ck[i][2] - iz) ** 2
                    D = math.sqrt(dc2 + ds2 * (m / interval) ** 2)
                    if D < distance[ix][iy][iz]:
                        distance[ix][iy][iz] = D
                        label[ix][iy][iz] = i

    # update cluster center
    sum_ck = np.zeros((len(ck), 5))
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[1]):
            for k in range(0, a.shape[2]):
                sum_ck[label[i][j][k]][4] = sum_ck[label[i][j][k]][4] + 1
                sum_ck[label[i][j][k]][:4] = sum_ck[label[i][j][k]][:4] + np.array([i, j, k, a[i][j][k]])
    for i in range(len(ck)):
        if redundant_flag[i]:
            continue
        if sum_ck[i][4] == 0:
            redundant_flag[i] = True
            continue
        assert sum_ck[i][4] > 0
        sum_ck[i][:4] = sum_ck[i][:4] / sum_ck[i][4]
        ck[i] = sum_ck[i][:4]
    return distance, label, ck, redundant_flag


@jit(nopython=True)
def renumber(redundant_flag, label):
    reduce_index = np.zeros(len(redundant_flag))
    cnt = 0
    for i in range(len(reduce_index)):
        if redundant_flag[i]:
            cnt += 1
        else:
            reduce_index[i] = cnt
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            for k in range(label.shape[2]):
                label[i][j][k] -= reduce_index[label[i][j][k]]
    return label


@jit(nopython=True)
def density_feature(a, feature_matrix, label, filters_num):
    min_val = np.min(a)
    max_val = np.max(a)
    width = (max_val - min_val) / 6
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                bin_num = min(int((a[i][j][k] - min_val) / width), 5)  # normalize
                feature_matrix[filters_num + bin_num][label[i][j][k]] += 1
    return feature_matrix


@jit(nopython=True)
def generate_saliency_map(a, label, supervoxel_saliency, pick_num):
    min_saliency = np.min(supervoxel_saliency)
    max_saliency = np.max(supervoxel_saliency)
    t = (min_saliency + max_saliency) / 2  # threshold
    if pick_num is not None:
        unqiue_saliency = np.unique(supervoxel_saliency)
        t = heapq.nlargest(pick_num, unqiue_saliency)[-1]
    print('min=', min_saliency, 'max=', max_saliency, 'threshold=', t)

    for i in range(len(supervoxel_saliency)):
        if supervoxel_saliency[i] < t:
            supervoxel_saliency[i] = min_saliency
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for k in range(a.shape[2]):
                a[i][j][k] = supervoxel_saliency[label[i][j][k]]
    return a


def particle_picking(a, saliency_map, ref_saliency_max, ref_saliency_min):
    """
    a : the original tomogram
    saliency_map: the modified output volume data a from step 5
    ref_saliency_max: the maximum saliency value
    ref_saliency_min: the minimun salienct value
    """
    n = 0  # subtom number iterator

    dif = 8  # half of the frame size

    for i in range(saliency_map.shape[0]):
        for j in range(saliency_map.shape[1]):
            for k in range(saliency_map.shape[2]):
                # finding the saliency value that is greater or above 90% of the max saliency value
                if saliency_map[i][j][k] >= 0.9 * ref_saliency_max:
                    # pass if it is on the edge
                    # TODO: edge case handle
                    if (i - dif < 0 or i + dif > saliency_map.shape[0] or
                            j - dif < 0 or j + dif > saliency_map.shape[1] or
                            k - dif < 0 or k + dif > saliency_map.shape[2]):
                        pass
                    else:
                        # frame the 3d subarray from the original tomogram
                        subtom = a[i - dif:i + dif, j - dif:j + dif, k - dif:k + dif]

                        print('x axis starting and end:', i - dif, "and", i + dif)
                        print('y axis starting and end:', j - dif, "and", j + dif)
                        print('z axis starting and end:', k - dif, "and", k + dif)
                        print("the dimension of subtom is", subtom.shape[0], "by", subtom.shape[1], "by",
                              subtom.shape[2])

                        n += 1
                        namemrc = "Desktop/result/saliency_map_subtomograms" + str(n) + ".mrc"
                        namepng = "Desktop/result/saliency_map_subtomograms" + str(n) + ".png"
                        io_file.put_mrc_data(subtom, namemrc)  # save as the mrc file
                        img = (subtom[:, :, int(subtom.shape[2] / 2)]).copy()
                        plt.imsave(namepng, img, cmap='gray')

                        # update the saliency map by filling the "cut" subtomogram matrices with minimum saliency value
                        saliency_map[i - dif:i + dif][j - dif:j + dif][k - dif:k + dif] = ref_saliency_min


def gabor_feature_single_job(a, filters, fm_i, label, cluster_center_number, save_flag):
    # convolution
    start_time = time.time()
    # b=SN.correlate(a,filters[i]) # too slow
    b = signal.correlate(a, filters[fm_i], mode='same')
    end_time = time.time()
    print('feature %d done (%f s)' % (fm_i, end_time - start_time))

    # show Gabor filter output
    if save_flag:
        img = (b[:, :, int(a.shape[2] / 2)]).copy()
        # save fig
        plt.imsave('Desktop/result/gabor_output(%d).png' % fm_i, img, cmap='gray')

    # generate feature vector
    start_time = time.time()
    result = generate_feature_vector(b=b, label=label, cluster_center_number=cluster_center_number)
    end_time = time.time()
    print('feature vector %d done (%f s)' % (fm_i, end_time - start_time))
    return fm_i, result


@jit(nopython=True)
def generate_feature_vector(b, label, cluster_center_number):
    result = np.array([0] * cluster_center_number)
    # sum_f = np.array((cluster_center_number, 2), 0)
    sum_f = np.array([[0 for i in range(2)] for j in range(cluster_center_number)])
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            for k in range(0, b.shape[2]):
                sum_f[label[i][j][k]][1] = sum_f[label[i][j][k]][1] + 1
                sum_f[label[i][j][k]][0] = sum_f[label[i][j][k]][0] + b[i][j][k]
    for i in range(cluster_center_number):
        assert sum_f[i][1] > 0
        result[i] = sum_f[i][0] / sum_f[i][1]
    return result


if __name__ == "__main__":
    # file path
    path = input("Enter data path: ")
    from aitom.io import mrcfile_proxy

    a = mrcfile_proxy.read_data(path)
    print("file has been read, shape is", a.shape)
    start_time = time.time()
    saliency_detection(a=a, gaussian_sigma=2.5, gabor_sigma=14.0, gabor_lambda=13.0, cluster_center_number=10000,
                       multiprocessing_num=0, pick_num=1000, save_flag=True)
    end_time = time.time()
    print('saliency detection takes', end_time - start_time, 's')

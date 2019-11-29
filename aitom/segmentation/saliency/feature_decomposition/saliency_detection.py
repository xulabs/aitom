import scipy.ndimage as SN
import numpy as np
import aitom.io.file as io_file
import matplotlib.pyplot as plt
import time
import math
import numpy.linalg
from numba import jit
from scipy.spatial.distance import cdist
from scipy import signal


'''
saliency detection
parameters:     
a: volume data      gaussian_sigma: sigma for de-noise      gabor_sigma/gabor_lambda: sigma/lambda for Gabor filter   
cluster_center_number: initial number of cluster centers        save_flag: set True to save results

return:     saliency map, the same shape as a
'''
def saliency_detection(a, gaussian_sigma, gabor_sigma, gabor_lambda, cluster_center_number, save_flag=False):
    # Step 1
    # Data Pre-processing
    a = SN.gaussian_filter(input=a, sigma=gaussian_sigma)  # de-noise
    print('sigma=', gaussian_sigma)
    if save_flag:
        img = (a[:, :, int(a.shape[2] / 2)]).copy()
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.savefig('./original.png')  # save fig

    # Step 2
    # Supervoxel over-segmentation
    N = a.shape[0] * a.shape[1] * a.shape[2]
    n = cluster_center_number
    ck = []  # cluster center [x y z g]
    interval = int(math.pow(N / n, 1.0 / 3))
    x = int(interval / 2)
    y = int(interval / 2)
    z = int(interval / 2)
    print('interval=%d' % interval)

    while (x < a.shape[0]) and (y < a.shape[1]) and (z < a.shape[2]):  # Initialization
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
    label = [[[0 for i in range(a.shape[2])] for i in range(a.shape[1])] for i in range(a.shape[0])]
    label = np.array(label)  # numba supports numpy array
    distance = [[[float('inf') for i in range(a.shape[2])] for i in range(a.shape[1])] for i in range(a.shape[0])]
    distance = np.array(distance)
    # label = np.zeros((a.shape[0], a.shape[1], a.shape[2])) # numba will report error
    # distance = np.full((a.shape[0], a.shape[1], a.shape[2]), np.inf)
    start_time = time.time()
    print('Supervoxel over-segmentation begins')
    ck = np.array(ck)
    redundant_flag = np.array([False] * len(ck))

    for number in range(10):  # 10 iterations suffices for most images
        b_time = time.time()
        print('\n%d of 10 iterations' % number)
        distance, label, ck = fast_SLIC(distance, label, ck, a, interval)
        # merge cluster centers
        ck_dist_min = interval / 2  # merge two cluster centers if the distance between them is less than ck_dist_min
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
    end_time = time.time()
    print('Supervoxel over-segmentation done,', end_time - start_time, 's')

    # save labels for Feature extraction
    labels_remove_num = sum(redundant_flag == True)
    labels = {}

    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[1]):
            for k in range(0, a.shape[2]):
                if label[i][j][k] in labels:
                    labels[label[i][j][k]].append([i, j, k])
                else:
                    labels[label[i][j][k]] = [[i, j, k]]
    assert labels_remove_num + len(labels) == len(ck)

    if save_flag:
        np.save('./labels', labels)
        img = (a[:, :, int(a.shape[2] / 2)]).copy()
        k = int(a.shape[2] / 2)
        draw_color = np.min(a)
        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if label[i][j][k] != label[i - 1][j][k] or label[i][j][k] != label[i + 1][j][k] or label[i][j][k] != \
                        label[i][j - 1][k] or label[i][j][k] != label[i][j + 1][k]:
                    img[i][j] = draw_color
        plt.axis('off')  # 不显示坐标轴
        plt.imshow(img, cmap='gray')
        plt.savefig('./SLIC.png')  # save fig

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
    feature_matrix = np.zeros((len(filters) + 6, len(labels)))  # Gabor filter bases features and 6 density features
    print('%d Gabor based features' % filters_num)

    print('Feature extraction begins')
    # 3D Gabor filter based features
    for i in range(len(filters)):
        # convolution
        start_time = time.time()
        # b=SN.correlate(a,filters[i]) # too slow
        b = signal.correlate(a, filters[i], mode='same')
        end_time = time.time()
        print('feature %d done (%f s)' % (i, end_time - start_time))

        # show Gabor filter output
        if save_flag:
            img = (b[:, :, int(a.shape[2] / 2)]).copy()
            plt.axis('off')  # 不显示坐标轴
            plt.imshow(img, cmap='gray')
            plt.savefig('./gabor_output(%d).png' % i)  # save fig

        # generate feature vector
        start_time = time.time()
        index_col = 0
        for key in labels:
            vox = labels[key]
            sum_vox = 0
            for j in range(len(vox)):
                sum_vox = sum_vox + b[vox[j][0], vox[j][1], vox[j][2]]
            # print('sum.type',type(sum)) <class 'numpy.float64'>
            sum_vox = sum_vox / len(vox)
            feature_matrix[i][index_col] = sum_vox
            index_col += 1
        # print(feature_matrix[i, 0:30])
        end_time = time.time()
        print('feature vector %d done (%f s)' % (i, end_time - start_time))
    print('3D Gabor filter based features done')

    # density features
    min_val = np.min(a)
    max_val = np.max(a)
    width = (max_val - min_val) / 6
    index_col = 0
    for key in labels:
        vox = labels[key]
        for j in vox:
            bin_num = min(int((a[j[0]][j[1]][j[2]] - min_val) / width), 5)  # normalize
            feature_matrix[filters_num + bin_num][index_col] += 1
        index_col += 1
    print('Density features done')

    if save_flag:
        np.save('./feature_matrix', feature_matrix)

    etime = time.time()
    print('Feature extraction done,', etime - stime, 's')

    # Step 4
    # RPCA
    start_time = time.time()
    print('RPCA begins')
    L, S = robust_pca(feature_matrix)
    end_time = time.time()
    print('RPCA done, ', end_time - start_time, 's')
    supervoxel_saliency = np.sum(S, axis=0) / S.shape[0]
    if save_flag:
        np.save('./supervoxel_saliency', supervoxel_saliency)

    # Step 5
    # Generate Saliency Map
    min_saliency = np.min(supervoxel_saliency)
    max_saliency = np.max(supervoxel_saliency)
    t = (min_saliency + max_saliency) / 2  # threshold
    print('min=', min_saliency, 'max=', max_saliency, 'threshold=', t)
    index_col = 0
    for key in labels:
        vox = labels[key]
        if supervoxel_saliency[index_col] < t:
            supervoxel_saliency[index_col] = min_saliency

        for j in vox:
            a[j[0]][j[1]][j[2]] = supervoxel_saliency[index_col]
        index_col += 1
        # print('sum.type',type(sum)) <class 'numpy.float64'>

    if save_flag:
        img = a[:, :, int(a.shape[2] / 2)].copy()
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        plt.savefig('./saliency_map.png')
        io_file.put_mrc_data(a, './saliency_map.mrc')
        print('saliency map saved')

    return a


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
    size = int(truncate*sigma + 0.5)
    '''
    for theta_x in np.arange(0, np.pi, np.pi / 4):
        for theta_y in np.arange(0, np.pi, np.pi / 4):
            for theta_z in np.arange(0, np.pi, np.pi / 4):
                if np.sum(np.abs([theta_x, theta_y, theta_z]) < 10e-8) < 2:
                    continue
                thetas = [theta_x, theta_y, theta_z]
                print(thetas)
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
    print("error =", error)
    return error <= initial_error * 10e-4


@jit(nopython=True)
def fast_SLIC(distance, label, ck, a, interval):
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
    sum = np.zeros((len(ck),5))
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[1]):
            for k in range(0, a.shape[2]):
                sum[label[i][j][k]][4]=sum[label[i][j][k]][4]+1
                sum[label[i][j][k]][:4]=sum[label[i][j][k]][:4]+np.array([i, j, k, a[i][j][k]])
    for i in range(len(ck)):
        if ck[i][3] == np.inf:
            continue
        assert sum[i][4]>0
        sum[i][:4]=sum[i][:4]/sum[i][4]
        ck[i]=sum[i][:4]
    return distance,label,ck


if __name__ == "__main__":
    path = './aitom_demo_single_particle_tomogram.mrc'  # file path
    mrc_header = io_file.read_mrc_header(path)
    a = io_file.read_mrc_data(path)  # volume data
    assert a.shape[0] > 0
    print("file has been read, shape is", a.shape)
    saliency_detection(a=a, gaussian_sigma=2.5, gabor_sigma=9.0, gabor_lambda=9.0, cluster_center_number=10000, save_flag=True)

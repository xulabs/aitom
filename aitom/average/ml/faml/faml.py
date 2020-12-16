import sys
import uuid
import os.path
import random
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift
import mrcfile as mrc
import pickle

import aitom.io.file as AIF
import aitom.io.db.lsm_db as TIDL
import aitom.parallel.multiprocessing.util as TPMU

import aitom.align.fast.util as align
import aitom.geometry.rotate as rotate
import aitom.geometry.ang_loc as ang_loc
import aitom.statistics.vol as stats
import aitom.image.io as AIIO
import aitom.image.vol.util as AIVU
import aitom.model.util as TMU

MULTIPROCESSING_WORKER_NUM = 8
'''
theta : {N : int,
         K : int,
         J : int,
         n : int,
         sigma_sq : float
         xi : float,
         alpha : 1*K nparray,
         A : K*J nparray,
         C : float
         trans_list : N*K array of list of transforms
         prob : N*K*number of transforms array of float
    }
X : {observed : N*J nparray,
     mask : N*J nparray}
phi : {q_rot : float,
       q_tilt : float,
       q_psi : float,
       q_x : float,
       q_y : float,
       q_z : float}
k : 1*N int array
'''
eps = np.finfo(np.float32).tiny


def fourier_transform(v):
    return fftshift(fftn(v))


def inv_fourier_transform(v):
    return ifftn(ifftshift(v)).real


def get_trans_list(k, i, theta):
    """
    Generates a list of possible tranformations for the purpose of integration
    In the Scheres paper the list is sampled from all possible transformations
    using adaptive sampling, but in our paper we only generate a small set
    corresponding to the best correlation scores in the Xu paper.
    """
    N = theta['N']
    K = theta['K']
    return theta['trans_list'][i][k]


def compute_trans_list(theta, img_data, use_voronoi):
    """
    Compute the list of optimal transformations for all (i, k) pairs using
    parallelism
    """

    N = theta['N']
    K = theta['K']
    n = theta['n']
    theta['trans_list'] = [[None for _ in range(K)] for _ in range(N)]
    # Compute the list of optimal transforms against all A_k's
    # for all i's in parallel
    tasks = {}
    for k_ in range(K):
        A_k = theta['A'][k_]
        for j, d in enumerate(img_data['dj']):
            t = dict()
            t['uuid'] = str(uuid.uuid4())
            t['module'] = 'aitom.average.ml.faml.faml'
            t['method'] = 'model_based_align_help'
            t['kwargs'] = {'img_db_path': img_data['db_path'],
                           'd': d,
                           'A_k': A_k,
                           'n': n,
                           'i': j,
                           'k': k_}
            tasks[t['uuid']] = t

    if len(sys.argv) > 1:
        rt_s = [_ for _ in TPRJB.run_iterator(
            tasks,
            redis_host=sys.argv[1],
            redis_port=6379,
            redis_password='2os43FR0Y1NVxAsy6k10A5to3oltsAl6vVeplZ9ktODQ88cs')]
    else:
        rt_s = [_ for _ in TPMU.run_iterator(
            tasks,
            worker_num=MULTIPROCESSING_WORKER_NUM)]

    for rt in rt_s:
        j = rt['result']['i']
        k_ = rt['result']['k']
        result = rt['result']['transforms']
        theta['trans_list'][j][k_] = result

    if use_voronoi:
        # Compute the weights for each configuration
        compute_voronoi_weights(theta)


def model_based_align_help(img_db_path, d, A_k, n, i, k):
    """
    Kernel function for computing the alignment between the data and class averages
    """
    X = get_image_db(img_db_path)

    v1 = inv_fourier_transform(X[d['v']])
    m1 = X[d['m']]

    v2 = inv_fourier_transform(A_k)
    m2 = TMU.sphere_mask([n, n, n])

    transforms = fast_align(v1, m1, v2, m2)
    result = []
    for item in transforms:
        phi = dict()
        phi['q_rot'] = item['ang'][0]
        phi['q_tilt'] = item['ang'][1]
        phi['q_psi'] = item['ang'][2]
        phi['q_x'] = item['loc'][0]
        phi['q_y'] = item['loc'][1]
        phi['q_z'] = item['loc'][2]
        result.append(phi)

    return {'i': i, 'k': k, 'transforms': result}


def compute_voronoi_weights(theta):
    """
    Compute the weights of each configuration
    by approximating the hypervolume of the voronoi region of each configuration
    in the 6-dimensional space of all configurations
    """
    N = theta['N']
    K = theta['K']
    n = theta['n']
    theta['voronoi'] = [[None for _ in range(K)] for _ in range(N)]

    tasks = {}
    for k in range(K):
        for i in range(N):
            trans_list = theta['trans_list'][i][k]

            t = dict()
            t['uuid'] = str(uuid.uuid4())
            t['module'] = 'aitom.geometry.volume.hypervolume.utils'
            t['method'] = 'voronoi_weights_6d'
            t['kwargs'] = {'phis': trans_list}
            t['i'] = i
            t['k'] = k
            tasks[t['uuid']] = t

    del i, k

    if len(sys.argv) > 1:
        rt_s = [_ for _ in TPRJB.run_iterator(
            tasks,
            redis_host=sys.argv[1],
            redis_port=6379,
            redis_password='2os43FR0Y1NVxAsy6k10A5to3oltsAl6vVeplZ9ktODQ88cs')]
    else:
        rt_s = [_ for _ in TPMU.run_iterator(
            tasks,
            worker_num=MULTIPROCESSING_WORKER_NUM)
                ]

    for rt in rt_s:
        i = tasks[rt['id']]['i']
        k = tasks[rt['id']]['k']
        theta['voronoi'][i][k] = rt['result']


def compute_prob(img_data, theta):
    """
    Compute the list of probabilities of k, phi and X_i given theta for all triples
    of (i, k, phi) with parallelism
    """
    dj = img_data['dj']
    try:
        assert theta['N'] == len(dj)
    except:
        raise Exception("Error in compute_prob: Inconsistent data dimensions!")

    N = theta['N']
    K = theta['K']
    n = theta['n']
    theta['ln_prob'] = [[None for _ in range(K)] for _ in range(N)]
    tasks = {}
    for k in range(K):
        A_k = theta['A'][k]
        for i, d in enumerate(img_data['dj']):
            trans_list = theta['trans_list'][i][k]
            t = dict()
            t['uuid'] = str(uuid.uuid4())
            t['module'] = 'aitom.average.ml.faml.faml'
            t['method'] = 'model_based_prob_help'
            t['kwargs'] = {
                'img_db_path': img_data['db_path'],
                'd': d,
                'A_k': A_k,
                'n': n,
                'i': i,
                'k': k,
                'J': theta['J'],
                'trans_list': trans_list,
                'sigma_sq': theta['sigma_sq'],
                'xi': theta['xi'],
                'alpha': theta['alpha']
            }
            tasks[t['uuid']] = t

    if len(sys.argv) > 1:
        rt_s = [_ for _ in TPRJB.run_iterator(
            tasks,
            redis_host=sys.argv[1],
            redis_port=6379,
            redis_password='2os43FR0Y1NVxAsy6k10A5to3oltsAl6vVeplZ9ktODQ88cs')]
    else:
        rt_s = [_ for _ in TPMU.run_iterator(
            tasks,
            worker_num=MULTIPROCESSING_WORKER_NUM)]

    for rt in rt_s:
        i = rt['result']['i']
        k = rt['result']['k']
        ln_prob_list = rt['result']['ln_prob']
        theta['ln_prob'][i][k] = ln_prob_list


def model_based_prob_help(A_k, trans_list, img_db_path, d, J, sigma_sq, n, i,
                          k, alpha, xi):
    """
    Computes the natural log of probability of k, phi and X^o_i given the model parameters
    note that some terms are omitted because they are canceled in the computation of P(k,phi|X_o,theta)
    """
    X = get_image_db(img_db_path)
    X_i = X[d['v']]
    w_i = X[d['m']]

    result = [None for _ in trans_list]
    for j, phi in enumerate(trans_list):
        ln_result = ln_p_xo(A_k=A_k, phi=phi, X_i=X_i, w_i=w_i, J=J, sigma_sq=sigma_sq,
                            n=n, k=k, i=i) + ln_p_k_phi(phi=phi, alpha=alpha, xi=xi)

        result[j] = ln_result

    return {'ln_prob': result, 'i': i, 'k': k}


def fast_align(v1, m1, v2, m2, max_l=36):
    """
    A wrapper to the fast alignment function in tomominer
    """
    angs = align.fast_rotation_align(v1, m1, v2, m2, max_l)
    a = align.translation_align_given_rotation_angles(v1, m1, v2, m2, angs)

    a = sorted(a, key=lambda _: (-_['score']))
    return a


def transform(phi, A_fourier, n, inv=False):
    """
    Generates the tomogram transformed by the rigid transformations specified by phi
    """
    A_real = inv_fourier_transform(A_fourier)

    ang = [phi['q_rot'], phi['q_tilt'], phi['q_psi']]
    loc = [phi['q_x'], phi['q_y'], phi['q_z']]

    if inv:
        ang, loc = ang_loc.reverse_transform_ang_loc(ang, loc)

    A_real_rot = rotate.rotate_pad_mean(A_real, angle=ang, loc_r=loc)

    result = fourier_transform(A_real_rot)
    return result


def inv_transform(phi, A_fourier, n):
    """
    Generates the tomogram transformed by the rigid transformations specified by the
    inverse of phi
    """
    return transform(phi, A_fourier, n, inv=True)


def inv_transform_mask(phi, w, n):
    """
    Generates the mask rotated by the angles specified by the inverse of phi
    """
    ang = [phi['q_rot'], phi['q_tilt'], phi['q_psi']]
    loc = [0.0, 0.0, 0.0]
    ang, loc = ang_loc.reverse_transform_ang_loc(ang, loc)
    w_rot = rotate.rotate_pad_zero(w, angle=ang, loc_r=loc)
    return w_rot


def p_k_phi_given_xo(k, phi_id, theta, i, use_voronoi):
    """
    Computes the probability of k and phi given X^o_i and theta
    This is kind of tricky to compute
    """
    K = theta['K']

    ln_num = theta['ln_prob'][i][k][phi_id]

    denom = 0.0
    for _k in range(K):
        phis = get_trans_list(k=_k, i=i, theta=theta)
        temp_sum = 0
        for j in range(len(phis)):
            if use_voronoi:
                dphi = theta['voronoi'][i][_k][j]
            else:
                dphi = 1.0 / len(phis)
            power = theta['ln_prob'][i][_k][j] - ln_num
            temp_sum += theta['alpha'][_k] * np.exp(power) * dphi

        denom += temp_sum

    result = theta['alpha'][k] / denom
    return result


def ln_p_xo(A_k, phi, X_i, w_i, J, sigma_sq, n, i, k):
    """
    Computes the natural log of probability of X^o_i given k, phi, and theta
    note that some terms are omitted because they are canceled in the computation of P(k,phi|X_o,theta)
    """
    RA = transform(phi, A_k, n)

    sum_j = np.sum(np.square(np.absolute(RA - X_i) * w_i))
    ln_result = -(sum_j / (2 * sigma_sq))
    return ln_result


def ln_p_k_phi(phi, alpha, xi):
    """
    Computes the natural log of probability of the hidden parameters given model parameters
    note that some terms are omitted because they are canceled in the computation of P(k,phi|X_o,theta)
    """
    q_x = phi['q_x']
    q_y = phi['q_y']
    q_z = phi['q_z']
    r_sq = q_x ** 2 + q_y ** 2 + q_z ** 2
    ln_result = -(r_sq / (2 * xi ** 2))
    return ln_result


def update_a(img_data, theta, alpha, use_voronoi, reg=True):
    """
    Updates the A values in the parameters dictionary theta
    """
    dj = img_data['dj']
    compute_prob(img_data=img_data, theta=theta)

    K = theta['K']
    J = theta['J']
    N = theta['N']
    A_old = theta['A']
    n = theta['n']
    A_new = np.zeros([K, n, n, n], dtype=np.complex128)

    # Regularization terms
    reg_num = theta['theta_reg'] * np.sum(A_old, axis=0)
    reg_denom = theta['theta_reg'] * K

    tasks = {}
    for k in range(K):
        for i, d in enumerate(img_data['dj']):
            t = {}
            t['uuid'] = str(uuid.uuid4())
            t['module'] = 'aitom.average.ml.faml.faml'
            t['method'] = 'update_a_help'
            t['kwargs'] = {
                'img_db_path': img_data['db_path'],
                'd': d,
                'i': i,
                'theta': theta,
                'k': k,
                'use_voronoi': use_voronoi
            }
            tasks[t['uuid']] = t

    if len(sys.argv) > 1:
        rt_s = [_ for _ in TPRJB.run_iterator(
            tasks,
            redis_host=sys.argv[1],
            redis_port=6379,
            redis_password='2os43FR0Y1NVxAsy6k10A5to3oltsAl6vVeplZ9ktODQ88cs')]
    else:
        rt_s = [_ for _ in TPMU.run_iterator(
            tasks,
            worker_num=MULTIPROCESSING_WORKER_NUM)]

    for rt in rt_s:
        k = rt['result']['k']
        A_new[k] += rt['result']['result']

    for k in range(K):
        if reg and theta['theta_reg'] > 0:
            A_new[k] += reg_num
            A_new[k] /= alpha[k] * N + reg_denom
        else:
            A_new[k] /= alpha[k] * N

    return A_new


def update_a_help(img_db_path, i, theta, k, d, use_voronoi):
    """
    Kernel function for updating A
    """
    X = get_image_db(img_db_path)
    w_i = X[d['m']]
    X_i = X[d['v']]

    K = theta['K']
    J = theta['J']
    N = theta['N']
    A_old = theta['A']
    n = theta['n']
    result = np.zeros([n, n, n], dtype=np.complex128)
    phis = get_trans_list(k=k, i=i, theta=theta)
    temp_sum = np.zeros([n, n, n], dtype=np.complex128)

    for j, phi in enumerate(phis):
        if use_voronoi:
            dphi = theta['voronoi'][i][k][j]
        else:
            dphi = 1.0 / len(phis)

        prob = p_k_phi_given_xo(k=k, phi_id=j, theta=theta, i=i, use_voronoi=use_voronoi)
        inv_mask = inv_transform_mask(phi, w_i, n)
        R_inv_X = inv_transform(phi, X_i, n)
        temp = inv_mask * R_inv_X + (np.ones(inv_mask.shape) - inv_mask) * A_old[k]
        temp_sum += prob * temp * dphi

    result = temp_sum

    return {'result': result, 'k': k}


def update_sigma(img_data, theta, use_voronoi, reg=True):
    """
    Updates the sigma_sq values in the parameters dictionary theta
    """
    compute_prob(img_data=img_data, theta=theta)

    K = theta['K']
    J = theta['J']
    N = theta['N']
    A = theta['A']
    n = theta['n']
    sigma_sq_old = theta['sigma_sq']

    # Regularization term
    reg_term = 0
    if reg and theta['theta_reg'] > 0:
        for k in range(K):
            for l in range(K):
                reg_term += np.sum(np.square(np.absolute(A[k] - A[l])))
        reg_term *= theta['theta_reg']

    tasks = {}
    for i, d in enumerate(img_data['dj']):
        t = dict()
        t['uuid'] = str(uuid.uuid4())
        t['module'] = 'aitom.average.ml.faml.faml'
        t['method'] = 'update_sigma_help'
        t['kwargs'] = {
            'img_db_path': img_data['db_path'],
            'i': i,
            'theta': theta,
            'd': d,
            'use_voronoi': use_voronoi
        }
        tasks[t['uuid']] = t

    if len(sys.argv) > 1:
        rt_s = [_ for _ in TPRJB.run_iterator(
            tasks,
            redis_host=sys.argv[1],
            redis_port=6379,
            redis_password='2os43FR0Y1NVxAsy6k10A5to3oltsAl6vVeplZ9ktODQ88cs')]
    else:
        rt_s = [_ for _ in TPMU.run_iterator(
            tasks,
            worker_num=MULTIPROCESSING_WORKER_NUM)]

    sigma_sq_new = 0

    for rt in rt_s:
        sigma_sq_new += rt['result']['result']

    if reg:
        sigma_sq_new += reg_term

    return sigma_sq_new / (N * J)


def update_sigma_help(img_db_path, i, theta, d, use_voronoi):
    """
    Kernel function for updating sigma_sq
    """
    X = get_image_db(img_db_path)
    w_i = X[d['m']]
    X_i = X[d['v']]

    K = theta['K']
    N = theta['N']
    A = theta['A']
    n = theta['n']
    sigma_sq_old = theta['sigma_sq']

    result = 0

    for k in range(K):
        phis = get_trans_list(k=k, i=i, theta=theta)
        A_k = A[k]
        temp_sum = 0
        for j, phi in enumerate(phis):
            if use_voronoi:
                dphi = theta['voronoi'][i][k][j]
            else:
                dphi = 1.0 / len(phis)

            prob = p_k_phi_given_xo(k=k, phi_id=j, theta=theta, i=i, use_voronoi=use_voronoi)
            RA = transform(phi, A_k, n)
            first_term = np.square(np.absolute(RA - X_i)) * w_i
            second_term = (np.ones(w_i.shape) - w_i) * sigma_sq_old
            temp_sum += prob * np.sum(first_term + second_term) * dphi

        result += temp_sum
    return {'result': result}


def update_alpha(img_data, theta, use_voronoi):
    """
    Updates the alpha values in the parameters dictionary theta
    """
    compute_prob(img_data=img_data, theta=theta)
    K = theta['K']
    N = theta['N']

    probs = np.zeros([N, K])

    tasks = {}
    for k in range(K):
        for i, d in enumerate(img_data['dj']):
            t = {}
            t['uuid'] = str(uuid.uuid4())
            t['module'] = 'aitom.average.ml.faml.faml'
            t['method'] = 'update_alpha_help'
            t['kwargs'] = {
                'i': i,
                'theta': theta,
                'k': k,
                'use_voronoi': use_voronoi
            }
            tasks[t['uuid']] = t

    if len(sys.argv) > 1:
        rt_s = [_ for _ in TPRJB.run_iterator(
            tasks,
            redis_host=sys.argv[1],
            redis_port=6379,
            redis_password='2os43FR0Y1NVxAsy6k10A5to3oltsAl6vVeplZ9ktODQ88cs')]
    else:
        rt_s = [_ for _ in TPMU.run_iterator(
            tasks,
            worker_num=MULTIPROCESSING_WORKER_NUM)]

    for rt in rt_s:
        k = rt['result']['k']
        i = rt['result']['i']
        probs[i][k] = rt['result']['result']

    alpha_new = np.sum(probs, axis=0) / N

    theta['predictions'] = np.argmax(probs, axis=1)

    return alpha_new


def update_alpha_help(i, theta, k, use_voronoi):
    """
    Kernel function for updating alpha
    """
    phis = get_trans_list(k=k, i=i, theta=theta)
    temp_sum = 0.0
    for j, phi in enumerate(phis):
        if use_voronoi:
            dphi = theta['voronoi'][i][k][j]
        else:
            dphi = 1.0 / len(phis)

        prob = p_k_phi_given_xo(k=k, phi_id=j, theta=theta, i=i, use_voronoi=use_voronoi)
        temp_sum += prob * dphi
    return {'i': i, 'k': k, 'result': temp_sum}


def update_xi(img_data, theta, use_voronoi):
    """
    Updates the xi values in the parameters dictionary theta
    """
    compute_prob(img_data=img_data, theta=theta)
    K = theta['K']
    J = theta['J']
    N = theta['N']

    xi_sq_new = 0

    tasks = {}
    for i in range(N):
        t = dict()
        t['uuid'] = str(uuid.uuid4())
        t['module'] = 'aitom.average.ml.faml.faml'
        t['method'] = 'update_xi_help'
        t['kwargs'] = {'i': i, 'theta': theta, 'use_voronoi': use_voronoi}
        tasks[t['uuid']] = t

    if len(sys.argv) > 1:
        rt_s = [_ for _ in TPRJB.run_iterator(
            tasks,
            redis_host=sys.argv[1],
            redis_port=6379,
            redis_password='2os43FR0Y1NVxAsy6k10A5to3oltsAl6vVeplZ9ktODQ88cs')]
    else:
        rt_s = [_ for _ in TPMU.run_iterator(
            tasks,
            worker_num=MULTIPROCESSING_WORKER_NUM)]

    for rt in rt_s:
        xi_sq_new += rt['result']['result']

    return max(eps, (xi_sq_new / (3 * N)) ** 0.5)


def update_xi_help(i, theta, use_voronoi):
    """
    Kernel function for updating xi
    """
    K = theta['K']
    N = theta['N']

    result = 0
    for k in range(K):
        phis = get_trans_list(k=k, i=i, theta=theta)
        temp_sum = 0
        for j, phi in enumerate(phis):

            if use_voronoi:
                dphi = theta['voronoi'][i][k][j]
            else:
                dphi = 1.0 / len(phis)

            q_x = phi['q_x']
            q_y = phi['q_y']
            q_z = phi['q_z']
            r_sq = q_x ** 2 + q_y ** 2 + q_z ** 2
            prob = p_k_phi_given_xo(k=k, phi_id=j, theta=theta, i=i, use_voronoi=use_voronoi)
            temp_sum += prob * r_sq * dphi
        result += temp_sum
    return {'result': result}


def EM(img_data, K, iteration, path, snapshot_interval=5,
       reg=False, use_voronoi=True):
    """
    The main estimation-maximization algorithm
    """
    np.seterr(all='ignore')
    X = get_image_db(img_data['db_path'])
    dj = img_data['dj']

    N = len(dj)
    n_x, n_y, n_z = X[dj[0]['v']].shape

    theta = dict()
    theta['N'] = N
    theta['J'] = n_x * n_y * n_z
    theta['n'] = n_x
    theta['K'] = K
    # Proportional to the radius of the image
    theta['xi'] = theta['n']
    # We need to initialize this later
    theta['A'] = np.zeros([K, n_x, n_y, n_z], dtype=np.complex128)
    theta['alpha'] = np.ones([K], dtype=np.float_) / K
    theta['trans_list'] = None
    theta['predictions'] = np.zeros([N])

    # Print relavent information
    print("Running model based alignment: N=%d, K=%d, dimensions=(%d,%d,%d)" %
          (N, K, n_x, n_y, n_z))
    if reg:
        print("With regularization")
    else:
        print("Without regularization")
    if use_voronoi:
        print("With voronoi weights")
    else:
        print("Without voronoi weights")

    # Regularization
    reg_step = (float(N) / K ** 2) / 2
    theta['theta_reg'] = 5 * reg_step if reg else 0

    # Sample K random data points from the set to initialize A
    indices = np.random.permutation(N)
    num_models = [0 for _ in range(K)]
    k = 0
    for i in range(N):
        theta['A'][k] += X[dj[indices[i]]['v']] * X[dj[indices[i]]['m']]
        num_models[k] += 1
        k = (k + 1) % K

    for k in range(K):
        theta['A'][k] /= num_models[k]

    # Get a random A_k and a random X_i and calculate sum_j to get sigma_sq
    k = np.random.randint(K)
    i = np.random.randint(N)
    sum_j = np.sum(
        np.square(np.absolute(theta['A'][k] - X[dj[i]['v']]) * X[dj[i]['m']]))
    theta['sigma_sq'] = sum_j / theta['J']
    print("Sigma_sq initialized to %d" % theta['sigma_sq'])

    checkpoint_dir = os.path.join(path, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    interval = snapshot_interval
    for i in range(iteration):
        checkpoint_file = os.path.join(checkpoint_dir, '%08d.pickle' % i)
        if os.path.exists(checkpoint_file):
            checkpoint_data = AIF.pickle_load(checkpoint_file)
            theta = checkpoint_data['theta']
            continue

        if i % interval == 0:
            output_images(theta, i, path=path)

        print("Running iteration %d" % (i + 1))
        # Update alpha before updating A
        compute_trans_list(theta=theta, img_data=img_data, use_voronoi=use_voronoi)

        alpha = update_alpha(img_data=img_data, theta=theta, use_voronoi=use_voronoi)
        print("Alpha updated! Alpha = ", end=' ')
        print(alpha.tolist())

        sigma_sq = update_sigma(img_data=img_data, theta=theta, reg=reg, use_voronoi=use_voronoi)
        print("Sigma updated! Sigma^2 = ", end=' ')
        print(sigma_sq)

        xi = update_xi(img_data=img_data, theta=theta, use_voronoi=use_voronoi)
        print("Xi updated! Xi = ", end=' ')
        print(xi)

        A = update_a(img_data=img_data, theta=theta, alpha=alpha, reg=reg, use_voronoi=use_voronoi)
        print("A updated! Average intensity of A = ", end=' ')
        print(np.average(A, (1, 2, 3)))

        theta['alpha'] = alpha
        theta['sigma_sq'] = sigma_sq
        theta['xi'] = xi
        theta['A'] = A
        # Since we changed the models A, the list of optimal transforms
        # needs to be re-calculated
        theta['trans_list'] = None
        theta['pred'] = None

        # Decrease the regularization coefficient
        if reg and theta['theta_reg'] > 0:
            theta['theta_reg'] -= reg_step
            theta['theta_reg'] = max(0, theta['theta_reg'])

        try:
            assert not os.path.exists(checkpoint_file)
        except:
            raise Exception("Checkpoint file already exists!")
        AIF.pickle_dump({'theta': theta}, checkpoint_file)

    print_prediction_results(theta, img_data)
    output_images(theta, iteration, path=path)
    print("Prediction from model: ", end=' ')
    print(theta['predictions'])
    return theta


def test_EM_real_data(img_data, iteration, K, snapshot_interval, path,
                      reg=False, use_voronoi=True):
    dj = img_data['dj']

    observed = []
    masks = []
    i = 0
    save_interval = 50

    X = get_image_db(img_data['db_path'])

    N = len(dj)
    size = X[dj[0]['v']].shape[0]

    EM(img_data=img_data, K=K, iteration=iteration,
       snapshot_interval=snapshot_interval, path=path,
       reg=reg, use_voronoi=use_voronoi)


def get_image_db(img_db_path):
    """
    Retrive image from the datebase path
    """
    X = TIDL.LSM(img_db_path, readonly=True)
    return X


def read_model(i, path, wedge_ang1, wedge_ang2, wedge_dir):
    """
    Read subtomograms from mrc files
    """
    import aitom.image.vol.wedge.util as W
    result = {'i': i}
    f = mrc.open(path)
    size = f.data.shape[0]
    result['observed'] = fourier_transform(f.data)
    mask = W.wedge_mask(size=f.data.shape, ang1=wedge_ang1,
                        ang2=wedge_ang2, direction=wedge_dir)
    result['mask'] = mask
    return result


def output_images(theta, iteration, path):
    """
    Output the models as images of slices of the 3D geometry
    """
    n = theta['n']
    K = theta['K']

    if not os.path.exists(path):
        os.makedirs(path)

    for k in range(K):
        v = inv_fourier_transform(theta['A'][k])
        with open(path + str(k) + 'average.pickle', 'wb') as handle:
            pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path + 'theta.pickle', 'wb') as handle:
            pickle.dump(theta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        output_image(v, os.path.join(path, "A%04d_%04d.png") % (k, iteration))


def output_image(v, path):
    """
    Saves the sliced image of model v to path
    """
    AIIO.save_png(AIVU.cub_img(v)['im'], path)


def print_prediction_results(theta, img_data, print_all=False):
    """
    Print prediction results from FSC correlation score
    """
    X = get_image_db(img_data['db_path'])
    dj = img_data['dj']
    N = len(dj)
    print("Prediction by fourier shell correlation:")
    for i in range(N):
        if print_all:
            for k in range(theta['K']):
                print(get_correlation_score(img_data, theta, X, i, k))
        else:
            print(get_correlation_score(img_data, theta, X, i))


def get_correlation_score(img_data, theta, img_db_path, d, k=None):
    """
    Returns FSC score between the kth average and the given subtomogram
    if k is unspecified, the best score is returned
    """
    # X = get_image_db(img_db_path)
    X = img_db_path
    n = theta['n']
    J = theta['J']
    K = theta['K']

    # v1 = inv_fourier_transform(X[d['v']])
    # m1 = X[d['m']]
    dj = img_data['dj']
    v1 = inv_fourier_transform(X[dj[d]['v']])
    m1 = X[dj[d]['m']]
    if k is not None:
        v2 = inv_fourier_transform(theta['A'][k])
        # m2 = np.ones((n, n, n))
        m2 = TMU.sphere_mask([n, n, n])
        item = fast_align(v1, m1, v2, m1)[0]
        best_ang = item['ang']
        best_loc = item['loc']
        A_real_pred = v2
        k_pred = k
    else:
        best_ang = None
        best_loc = None
        best_score = None
        A_real_pred = None
        k_pred = None

        for k in range(K):
            v2 = inv_fourier_transform(theta['A'][k])
            # m2 = np.ones((n, n, n))
            m2 = TMU.sphere_mask([n, n, n])

            transforms = fast_align(v1, m1, v2, m2)
            item = transforms[0]
            score = item['score']

            if best_score is None or score > best_score:
                best_score = score
                best_ang = item['ang']
                best_loc = item['loc']
                A_real_pred = v2
                k_pred = k

    A_aligned = rotate.rotate_pad_mean(A_real_pred, angle=best_ang, loc_r=best_loc)

    return "Model", dj[d]['v'], k_pred, stats.fsc(v1, A_aligned)

"""
Weighted Principal Component Analysis using Expectation Maximization

Classic PCA is great but it doesn't know how to handle noisy or missing
data properly.  This module provides Weighted Expectation Maximization PCA,
an iterative method for solving PCA while properly weighting data.
Missing data is simply the limit of weight=0.

Given data[nobs, nvar] and weights[nobs, nvar],

    m = empca(data, weights, options...)

Returns a Model object m, from which you can inspect the eigenvectors,
coefficients, and reconstructed model, e.g.

    pylab.plot( m.eigvec[0] )
    pylab.plot( m.data[0] )
    pylab.plot( m.model[0] )
    
For comparison, two alternate methods are also implemented which also
return a Model object:

    m = lower_rank(data, weights, options...)
    m = classic_pca(data)  #- but no weights or even options...
    
Stephen Bailey, Spring 2012
"""

import numpy as N
import sys
from scipy.sparse import dia_matrix
import scipy.sparse.linalg
import math


class Model(object):
    """
    A wrapper class for storing data, eigenvectors, and coefficients.
    
    Returned by empca() function.  Useful member variables:
      Inputs: 
        - eigvec [nvec, nvar]
        - data   [nobs, nvar]
        - weights[nobs, nvar]
      
      Calculated from those inputs:
        - coeff  [nobs, nvec] - coeffs to reconstruct data using eigvec
        - model  [nobs, nvar] - reconstruction of data using eigvec,coeff
    
    Not yet implemented: eigenvalues, mean subtraction/bookkeeping
    """

    def __init__(self, eigvec, data, weights):
        """
        Create a Model object with eigenvectors, data, and weights.
        
        Dimensions:
          - eigvec [nvec, nvar]  = [k, j]
          - data   [nobs, nvar]  = [i, j]
          - weights[nobs, nvar]  = [i, j]
          - coeff  [nobs, nvec]  = [i, k]        
        """
        self.eigvec = eigvec
        self.nvec = eigvec.shape[0]

        self.set_data(data, weights)

    def set_data(self, data, weights):
        """
        Assign a new data[nobs,nvar] and weights[nobs,nvar] to use with
        the existing eigenvectors.  Recalculates the coefficients and
        model fit.
        """
        self.data = data
        self.weights = weights

        self.nobs = data.shape[0]
        self.nvar = data.shape[1]
        self.coeff = N.zeros((self.nobs, self.nvec))
        self.model = N.zeros(self.data.shape)

        # - Calculate degrees of freedom
        ii = N.where(self.weights > 0)
        self.dof = self.data[ii].size - self.eigvec.size - self.nvec * self.nobs

        # - Cache variance of unmasked data
        self._unmasked = ii
        self._unmasked_data_var = N.var(self.data[ii])

        self.solve_coeffs()

    def solve_coeffs(self):
        """
        Solve for c[i,k] such that data[i] ~= Sum_k: c[i,k] eigvec[k]
        """
        for i in range(self.nobs):
            # - Only do weighted solution if really necessary
            if N.any(self.weights[i] != self.weights[i, 0]):
                self.coeff[i] = _solve(self.eigvec.T, self.data[i], self.weights[i])
            else:
                self.coeff[i] = N.dot(self.eigvec, self.data[i])

        self.solve_model()

    def solve_eigenvectors(self, smooth=None):
        """
        Solve for eigvec[k,j] such that data[i] = Sum_k: coeff[i,k] eigvec[k]
        """

        # - Utility function; faster than numpy.linalg.norm()
        def norm(x):
            return N.sqrt(N.dot(x, x))

        # - Make copy of data so we can modify it
        data = self.data.copy()

        # - Solve the eigenvectors one by one
        for k in range(self.nvec):

            # - Can we compact this loop into numpy matrix algebra?
            c = self.coeff[:, k]
            for j in range(self.nvar):
                w = self.weights[:, j]
                x = data[:, j]
                # self.eigvec[k, j] = c.dot(w*x) / c.dot(w*c)
                # self.eigvec[k, j] = w.dot(c*x) / w.dot(c*c)
                cw = c * w
                self.eigvec[k, j] = x.dot(cw) / c.dot(cw)

            if smooth is not None:
                self.eigvec[k] = smooth(self.eigvec[k])

            # - Remove this vector from the data before continuing with next
            # ? Alternate: Resolve for coefficients before subtracting?
            # - Loop replaced with equivalent N.outer(c,v) call (faster)
            # for i in range(self.nobs):
            #     data[i] -= self.coeff[i,k] * self.eigvec[k]

            data -= N.outer(self.coeff[:, k], self.eigvec[k])

            # - Renormalize and re-orthogonalize the answer
        self.eigvec[0] /= norm(self.eigvec[0])
        for k in range(1, self.nvec):
            for kx in range(0, k):
                c = N.dot(self.eigvec[k], self.eigvec[kx])
                self.eigvec[k] -= c * self.eigvec[kx]

            self.eigvec[k] /= norm(self.eigvec[k])

        # - Recalculate model
        self.solve_model()

    def solve_model(self):
        """
        Uses eigenvectors and coefficients to model data
        """
        for i in range(self.nobs):
            self.model[i] = self.eigvec.T.dot(self.coeff[i])

    def chi2(self):
        """
        Returns sum( (model-data)^2 / weights )
        """
        delta = (self.model - self.data) * N.sqrt(self.weights)
        return N.sum(delta ** 2)

    def rchi2(self):
        """
        Returns reduced chi2 = chi2/dof
        """
        return self.chi2() / self.dof

    def _model_vec(self, i):
        """Return the model using just eigvec i"""
        return N.outer(self.coeff[:, i], self.eigvec[i])

    def R2vec(self, i):
        """
        Return fraction of data variance which is explained by vector i.

        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """

        d = self._model_vec(i) - self.data
        return 1.0 - N.var(d[self._unmasked]) / self._unmasked_data_var

    def R2(self, nvec=None):
        """
        Return fraction of data variance which is explained by the first
        nvec vectors.  Default is R2 for all vectors.
        
        Notes:
          - Does *not* correct for degrees of freedom.
          - Not robust to data outliers.
        """
        if nvec is None:
            mx = self.model
        else:
            mx = N.zeros(self.data.shape)
            for i in range(nvec):
                mx += self._model_vec(i)

        d = mx - self.data

        # - Only consider R2 for unmasked data
        return 1.0 - N.var(d[self._unmasked]) / self._unmasked_data_var


def _random_orthonormal(nvec, nvar, seed=1):
    """
    Return array of random orthonormal vectors A[nvec, nvar] 

    Doesn't protect against rare duplicate vectors leading to 0s
    """

    if seed is not None:
        N.random.seed(seed)

    A = N.random.normal(size=(nvec, nvar))
    for i in range(nvec):
        A[i] /= N.linalg.norm(A[i])

    for i in range(1, nvec):
        for j in range(0, i):
            A[i] -= N.dot(A[j], A[i]) * A[j]
            A[i] /= N.linalg.norm(A[i])

    return A


def _solve(A, b, w):
    """
    Solve Ax = b with weights w; return x
    
    A : 2D array
    b : 1D array length A.shape[0]
    w : 1D array same length as b
    """

    # - Apply weights
    # nvar = len(w)
    # W = dia_matrix((w, 0), shape=(nvar, nvar))
    # bx = A.T.dot( W.dot(b) )
    # Ax = A.T.dot( W.dot(A) )

    b = A.T.dot(w * b)
    A = A.T.dot((A.T * w).T)

    if isinstance(A, scipy.sparse.spmatrix):
        x = scipy.sparse.linalg.spsolve(A, b)
    else:
        # x = N.linalg.solve(A, b)
        x = N.linalg.lstsq(A, b, rcond=-1)[0]

    return x


# -------------------------------------------------------------------------

def empca(data, weights=None, niter=25, nvec=5, smooth=0, randseed=1):
    """
    Iteratively solve data[i] = Sum_j: c[i,j] p[j] using weights
    
    Input:
      - data[nobs, nvar]
      - weights[nobs, nvar]
      
    Optional:
      - niter    : maximum number of iterations
      - nvec     : number of model vectors
      - smooth   : smoothing length scale (0 for no smoothing)
      - randseed : random number generator seed; None to not re-initialize
    
    Returns Model object
    """

    if weights is None:
        weights = N.ones(data.shape)

    if smooth > 0:
        smooth = SavitzkyGolay(width=smooth)
    else:
        smooth = None

    # - Basic dimensions
    nobs, nvar = data.shape
    assert data.shape == weights.shape

    # - degrees of freedom for reduced chi2
    ii = N.where(weights > 0)
    dof = data[ii].size - nvec * nvar - nvec * nobs

    # - Starting random guess
    eigvec = _random_orthonormal(nvec, nvar, seed=randseed)

    model = Model(eigvec, data, weights)
    model.solve_coeffs()

    # print "       iter    chi2/dof     drchi_E     drchi_M   drchi_tot       R2            rchi2"
    print("       iter        R2             rchi2")

    for k in range(niter):
        model.solve_coeffs()
        model.solve_eigenvectors(smooth=smooth)
        sys.stdout.write('\rEMPCA %2d/%2d  %15.8f %15.8f             ' % (k + 1, niter, model.R2(), model.rchi2()))
        sys.stdout.flush()

    # - One last time with latest coefficients
    model.solve_coeffs()

    print("R2:", model.R2())

    r2_vec = [model.R2vec(_) for _ in range(model.nvec)]
    # print "R2 cummulative", [sum(r2_vec[:_]) for _ in range(1, model.nvec + 1)]

    return model


def classic_pca(data, nvec=None):
    """
    Perform classic SVD-based PCA of the data[obs, var].
    
    Returns Model object
    """
    u, s, v = N.linalg.svd(data)
    if nvec is None:
        m = Model(v, data, N.ones(data.shape))
    else:
        m = Model(v[0:nvec], data, N.ones(data.shape))
    return m


def lower_rank(data, weights=None, niter=25, nvec=5, randseed=1):
    """
    Perform iterative lower rank matrix approximation of data[obs, var]
    using weights[obs, var].
    
    Generated model vectors are not orthonormal and are not
    rotated/ranked by ability to model the data, but as a set
    they are good at describing the data.
    
    Optional:
      - niter : maximum number of iterations to perform
      - nvec  : number of vectors to solve
      - randseed : rand num generator seed; if None, don't re-initialize
    
    Returns Model object
    """

    if weights is None:
        weights = N.ones(data.shape)

    nobs, nvar = data.shape
    P = _random_orthonormal(nvec, nvar, seed=randseed)
    C = N.zeros((nobs, nvec))
    ii = N.where(weights > 0)
    dof = data[ii].size - P.size - nvec * nobs

    print("iter     dchi2       R2             chi2/dof")

    oldchi2 = 1e6 * dof
    for blat in range(niter):
        # - Solve for coefficients
        for i in range(nobs):
            # - Convert into form b = A x
            b = data[i]  # - b[nvar]
            A = P.T  # - A[nvar, nvec]
            w = weights[i]  # - w[nvar]
            C[i] = _solve(A, b, w)  # - x[nvec]

        # - Solve for eigenvectors
        for j in range(nvar):
            b = data[:, j]  # - b[nobs]
            A = C  # - A[nobs, nvec]
            w = weights[:, j]  # - w[nobs]
            P[:, j] = _solve(A, b, w)  # - x[nvec]

        # - Did the model improve?
        model = C.dot(P)
        delta = (data - model) * N.sqrt(weights)
        chi2 = N.sum(delta[ii] ** 2)
        diff = data - model
        R2 = 1.0 - N.var(diff[ii]) / N.var(data[ii])
        dchi2 = (chi2 - oldchi2) / oldchi2  # - fractional improvement in chi2
        flag = '-' if chi2 < oldchi2 else '+'
        print('%3d  %9.3g  %15.8f %15.8f %s' % (blat, dchi2, R2, chi2 / dof, flag))
        oldchi2 = chi2

    # - normalize vectors
    for k in range(nvec):
        P[k] /= N.linalg.norm(P[k])

    m = Model(P, data, weights)
    print("R2:", m.R2())

    # - Rotate basis to maximize power in lower eigenvectors
    # --> Doesn't work; wrong rotation
    # u, s, v = N.linalg.svd(m.coeff, full_matrices=True)
    # eigvec = N.zeros(m.eigvec.shape)
    # for i in range(m.nvec):
    #     for j in range(s.shape[0]):
    #         eigvec[i] += v[i,j] * m.eigvec[j]
    # 
    #     eigvec[i] /= N.linalg.norm(eigvec[i])
    # 
    # m = Model(eigvec, data, weights)
    # print m.R2()

    return m


class SavitzkyGolay(object):
    """
    Utility class for performing Savitzky Golay smoothing
    
    Code adapted from http://public.procoders.net/sg_filter/sg_filter.py
    """

    def __init__(self, width, pol_degree=3, diff_order=0):
        self._width = width
        self._pol_degree = pol_degree
        self._diff_order = diff_order
        self._coeff = self._calc_coeff(width // 2, pol_degree, diff_order)

    def _calc_coeff(self, num_points, pol_degree, diff_order=0):

        """
        Calculates filter coefficients for symmetric savitzky-golay filter.
        see: http://www.nrbook.com/a/bookcpdf/c14-8.pdf
    
        num_points   means that 2*num_points+1 values contribute to the
                     smoother.
    
        pol_degree   is degree of fitting polynomial
    
        diff_order   is degree of implicit differentiation.
                     0 means that filter results in smoothing of function
                     1 means that filter results in smoothing the first 
                                                 derivative of function.
                     and so on ...
        """

        # setup interpolation matrix
        # ... you might use other interpolation points
        # and maybe other functions than monomials ....

        x = N.arange(-num_points, num_points + 1, dtype=int)
        monom = lambda x, deg: math.pow(x, deg)

        A = N.zeros((2 * num_points + 1, pol_degree + 1), float)
        for i in range(2 * num_points + 1):
            for j in range(pol_degree + 1):
                A[i, j] = monom(x[i], j)

        # calculate diff_order-th row of inv(A^T A)
        ATA = N.dot(A.transpose(), A)
        rhs = N.zeros((pol_degree + 1,), float)
        rhs[diff_order] = (-1) ** diff_order
        wvec = N.linalg.solve(ATA, rhs)

        # calculate filter-coefficients
        coeff = N.dot(A, wvec)

        return coeff

    def __call__(self, signal):
        """
        Applies Savitsky-Golay filtering
        """
        n = N.size(self._coeff - 1) / 2
        res = N.convolve(signal, self._coeff)
        return res[n:-n]


def _main():
    N.random.seed(1)
    nobs = 100
    nvar = 200
    nvec = 3
    data = N.zeros(shape=(nobs, nvar))

    # - Generate data
    x = N.linspace(0, 2 * N.pi, nvar)
    for i in range(nobs):
        for k in range(nvec):
            c = N.random.normal()
            data[i] += 5.0 * nvec / (k + 1) ** 2 * c * N.sin(x * (k + 1))

    # - Add noise
    sigma = N.ones(shape=data.shape)
    for i in range(nobs / 10):
        sigma[i] *= 5
        sigma[i, 0:nvar / 4] *= 5

    weights = 1.0 / sigma ** 2
    noisy_data = data + N.random.normal(scale=sigma)

    print("Testing empca")
    m0 = empca(noisy_data, weights, niter=20)

    print("Testing lower rank matrix approximation")
    m1 = lower_rank(noisy_data, weights, niter=20)

    print("Testing classic PCA")
    m2 = classic_pca(noisy_data)
    print("R2", m2.R2())

    try:
        import pylab as P
    except ImportError:
        print("pylab not installed; not making plots", file=sys.stderr)
        sys.exit(0)

    P.subplot(311)
    for i in range(nvec):
        P.plot(m0.eigvec[i])
    P.ylim(-0.2, 0.2)
    P.ylabel("EMPCA")
    P.title("Eigenvectors")

    P.subplot(312)
    for i in range(nvec):
        P.plot(m1.eigvec[i])
    P.ylim(-0.2, 0.2)
    P.ylabel("Lower Rank")

    P.subplot(313)
    for i in range(nvec):
        P.plot(m2.eigvec[i])
    P.ylim(-0.2, 0.2)
    P.ylabel("Classic PCA")

    P.show()


if __name__ == '__main__':
    _main()

'''
mxu:

empca is fast when extracting out only a small number of princile components
it allows integration of real space weights or masks that are specific to individual subtomograms

'''

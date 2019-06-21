#ifndef TOMO_SHT_HPP
#define TOMO_SHT_HPP

#include <armadillo>

/**
    @defgroup sht Spherical Harmonic Transform
    @{

    @section least Least-squares estimate of Spherical Harmonic Transform.

    Functions for Spherical Harmonic Transformations (SHT).  This code uses as a
    primary reference a paper by Martinec @cite martinec-1991

    The syntax used for the code is the same as in the article.  In particular
    the paper uses j instead of l for an index in the spherical harmonic
    functions, which is different then most references.  This is the only place
    in the code where j is used as the index of a spherical harmonic function.
    In the remainder of the code l is used instead, but I have left this as j
    to keep the code readable when comparing to the paper.

    This code is very inefficient.  An optimized version can be made following
    some of the optimizations described in the paper but not implemented.  In
    particular, this code does not use FFTs, or any symmetry.

    Additional speed benefits can be gained by moving to a quadrature method
    over non-equidistant points.  If we use Gaussian quadrature instead, we can
    precompute a large amount of the needed numbers, and accelerate the
    calculations.

    There are a large number of papers that should be looked at for
    improvements to this code.  Blais has several papers on methods to optimize
    SHT @cite blais_2005 @cite blais_2008.  Other accelerations and papers on
    optimizations I have not yet looked at include @cite potts_1998 @cite hupca_2010 
    @cite suda_2001 @cite choi_1999 @cite drake_2008 @cite mohlenkamp_1999

    @section quad Quadrature methods for SHT.

    Quadrature is based on the ability to compute a polynomial of degree 2N-1
    using N points over the surface of the sphere.

    For Gaussian quadrature (\f$g_j = cos^{-1}(x_j)\f$) where x_j are the roots
    of the Legendre polynomial of degree N on the interval [-1,1].

    Define weights \f[w_j = 2*(1-x_j)/((n+1)^2[P_{N+1}(x_j)]^2)\f]

    Then we can write the spherical harmonic coefficients exactly:

    \f[\alpha_n^m = \sum_{j=0}^{N} f(\phi_j, \hat(m)) P_n^m(\cos \theta) \sin(\phi_j) w_N(j)\f]

*/

/**
    Forward Spherical Harmonic Transform (SHT). 

    Given data on equally spaced angular grid, calculate the spherical harmonic
    coefficients to best approximate the function up to order j_max.

    @param f the data as a Nx2N matrix.  Data values over [0,pi]x[0,2pi]
    @param j_max maximum order of spherical harmonic to use in the expansion.

    @return The spherical harmonic coefficients.  The coefficients are in a
    square matrix.  Only lower-diagonal entries are filled in.  The (j,m) entry
    corresponds to the \f$Y_j^m\f$ entry.

*/
arma::cx_mat forward_sht(arma::mat f, unsigned int L);

/**
    Inverse Spherical Harmonic Transform.

    @note This is real valued only!!

    @param A_jm Coefficients of spherical harmonic functions, A lower triangular
    matrix with (j,m) entry corresponding to coefficient of harmonic \f$Y_j^m\f$. 
    @param N the dimension to expand data out to.  We will return the data
    sampled over a grid Nx2N
    
    @returns A grid of function values Nx2N

*/
arma::mat reverse_sht(arma::cx_mat A_jm, unsigned int nlat, unsigned int nlon);

/**
  @} // end group sht
*/
#endif

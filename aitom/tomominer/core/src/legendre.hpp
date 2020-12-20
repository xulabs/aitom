#ifndef TOMO_LEGENDRE_HPP
#define TOMO_LEGENDRE_HPP

#include <armadillo>

/** 
    @defgroup legendre_polynomials Associated Legendre polynomials
    @{

    @section alp Associated Legendre polynomials

    The Associated Legendre polynomial of degree \f$l\f$ and order \f$m\f$ is written as
    \f$P_l^m(x)\f$.

    http://en.wikipedia.org/wiki/Associated_Legendre_polynomials

    There are three recurrence relationships we use for the generation of the
    polynomials:

    \f[P(m,m) = (1-2m) \sqrt{1-x^2} P(m-1,m-1) \f]
    \f[P(m+1,m) = (2m+1) x P(m,m) \f]
    \f[P(l,m) = \frac{(2l-1) x P(l-1,m) - (l+m-1) P(l-2, m)}{(l-m)} \f]

    And we start with the identity:

    \f$P(0,0) = 1\f$

    We can build all the polynomials using this order:

    - Use first identity build the diagonal. (0,0), (1,1), (2,2), (3,3), ...
    - Use second identity  to build (m+1,m) for all m. (1,0), (2,1), (3,2), ...
    - Use third  identity to fill remaining elements (l,m) (2,0), (3,0), ..., (3,1),(4,1),..., (4,2), (5,2), ...

    @section alp_norm Including spherical harmonic normalization factors

    The Associated Legendre polynomials are used in spherical harmonics.  The spherical harmonic of degree \f$l\f$ and order \f$m\f$ is written as:

    \f[
    Y_l^m(\theta, \phi) = \sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}} P_l^m(\cos \theta) e^{im\phi}
    \f]

    For efficiency we will include the calculation of the normalization factor
    \f[ \sqrt{\frac{(2l+1)(l-m)!}{4\pi(l+m)!}}\f]  inside of the Legendre
    polynomial calculation.  This is also a more stable way to calculate the
    normalization factors. Building them recursively is more stable then
    generating them naively.

    We can modify our three recurrence relationships used to construct the polynomials to include the normalization factor.

    \f[ P(m,m) = -\sqrt{\frac{2m+1}{2m}} \sqrt{1-x^2} P(m-1,m-1) \f]
    \f[ P(m+1,m) = \sqrt{2m+3} x P(m,m) \f]
    \f[ P(l,m) =  \sqrt{\frac{(2l+1)*(2l-1)}{(l-m)*(l+m)}} x P(l-1,m) - \sqrt{\frac{(2l+1)(l-m-1)(l+m-1)}{(2l-3)(l-m)(l+m)}} P(l-2, m) \f]

    The initial condition is now:

    \f[P(0,0) = \frac{1}{\sqrt{4*pi}}\f]
    
    @section alp_neg Evaluation at negative orders

    The code below only provides the coefficients for orders m>=0.  The polynomial at negative \f$m\f$ can be found using the identity:

    \f[P_l^{-m} = (-1)^m \frac{(l-m)!}{(l+m)!} P_l^{m}  \f]

    for the regular Associated Legendre Polynomial, and 

    \f[P_l^{-m} = (-1)^mP_l^{m} \f]

    for the normalized polynomials  In the case of the normalized polynomials, we also have the property:
    
    \f[Y_l^{-m} = (-1)^mY_l^{m*} \f]

    where \f$Y^*\f$ is the conjugate of \f$Y\f$.  This is what is most often used, and the negative order Legendre polynomials are not used.

*/


/**
    Generate Associated Legendre polynomials evaluated at locations x.

    @param data A vector of locations we want to evaluate the associated Legendre polynomials at.
    @param l_max The maximum degree to use.
    @return The associated Legendre polynomial evaluated at each point for degrees l = 0,...,l_max and orders m = 0,...,l.  

    The returned cube P is indexed as:

    \f$P(l,m,i) = P_l^m(x_i)\f$

    @see legendre_polynomials, generate_normalized_legendre
*/
arma::cube generate_legendre(unsigned int l_max, const arma::vec &data);

/**
    Generate normalized associate Lengedre polynomials at locations x.

    @param data vector of evaluation values.
    @param l_max maximum order to evaluate up to.
    @returns P indexed by l,m,i for \f$P_l^m(x_i)\f$ for positive m only.

    The returned matrix P is indexed as:

    \f$P(l,m,i) = P_l^m(x_i)\f$

    @see legendre_polynomials, generate_legendre 
*/
arma::cube generate_normalized_legendre(unsigned int l_max, const arma::vec &data);

/**
  @} // end group legendre_polynomials
*/
#endif

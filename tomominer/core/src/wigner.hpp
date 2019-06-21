#ifndef TOMO_WIGNER_HPP
#define TOMO_WIGNER_HPP

#include <vector>

#include <armadillo>
/**
    @defgroup wigner Wigner small-d matrix generation
    @{

    Calculation of Wigner's small d-matrix.
    
    http://en.wikipedia.org/wiki/Wigner_D-matrix

    Calculation using the method described by Dachsel @cite dachsel_2006.

    The elements of the d-matrix are defined as:
    \f[
       d^j_{m'm}(\beta) = [(j+m')!(j-m')!(j+m)!(j-m)!]^{1/2} \sum_s \frac{(-1)^{m'-m+s}}{(j+m-s)!s!(m'-m+s)!(j-m'-s)!} \times \left(\cos\frac{\beta}{2}\right)^{2j+m-m'-2s}\left(\sin\frac{\beta}{2}\right)^{m'-m+2s} 
    \f]

    Where \f$j\f$ is the order of the matrix, and the elements at position \f$(m,m')\f$ run from index -j:j.

    We use several recursive formulas to build up the elements while preserving numerical stability.

*/

/**
    The Wigner D-matrix of order l, is a matrix of size (2*l+1)x(2*l+1).  With
    indexes -l:l in both dimensions.  

    @param theta Angle of rotation.  radians.
    @param max_l Maximum order of matrix to generate.
    @return List of wigner d-matrices, for order = 0, ..., max_l of sizes (1x1), (3x3), ..., ((2*max_l+1)x(2*max_l+1))
*/
std::vector<arma::mat> wigner_d(double theta, int max_l);

/**
  @}
*/
#endif

#ifndef TOMO_ALIGN_HPP
#define TOMO_ALIGN_HPP

#include <tuple>

//#include "boost/tuple/tuple.hpp"

#include <armadillo>


#include "arma_extend.hpp"
#include "dilate.hpp"
#include "fatal_error.hpp"
#include "fft.hpp"
#include "geometry.hpp"
#include "interpolation.hpp"
#include "io.hpp"
#include "rotate.hpp"
#include "sht.hpp"
#include "wigner.hpp"


/**
  Integrate the function using trapezoid rule.

  @param x the x values the function has been evaluated at.
  @param y the function values at the given x locations.
  @return approximate the integral of the function over the range of x, using the trapezoid rule.
*/
std::complex<double> trapz(const arma::vec &x, const arma::cx_vec &y);


/**
    @defgroup Constrained Correlation and Translational Alignment
    
    @{

    We calculate the constrained correlation coefficient (CCC) as defined by
    Förster @cite forster_2008.

    A subtomogram is a volume over a grid with values defined for T_i(x,y,z).
    A simple normalization subtracts the mean, and divides by the standard
    deviation:

    \f[
    T_i'(x,y,z) = \frac{T_i(x,y,z) - \bar{T_i}}{\sqrt{\sum_{x,y,z}(T_i(x,y,z) - \bar{T_i})^2}}
    \f]

    The normalization between two tomograms is defined by their element-wise product.
    \f[
    cor(T_i, T_j) = \sum_{x,y,z}T_i'(x,y,z) \cdot T_j'(x,y,z) 
    \f]

    There are several complications with our data however.  First our tomograms
    are not complete data.  They are sampled from a Fourier space which is
    missing a wedge.  This incomplete sampling is described by the wedge shaped
    region \f$\omega_i\f$.

    For two tomograms with different wedges and which align after rotations, we
    restrict our correlation calculations to the region for which both
    tomograms have been defined.

    \f[
    \Omega_{ij} = R(-\psi_i, -\phi_i, -\theta_i) \cdot R(-\psi_j, -\phi_j, -\theta_j) 
    \f]

    Where \f$R(-\psi_i, -\phi_i, -\theta_i)\f$ is the inverse rotation of
    defined by Euler angles rotating around \f$(\psi_i, \phi_i, \theta_i)\f$

    Also included is a mask to restrict the portions of the tomograms compared.

    Our new normalized tomogram is then:

    \f[ T_i' = \frac{ M(x,y,z) \cdot \mbox{FFT}^{-1}(\hat{T}_i \cdot \Omega_{ij}) - \bar{T}_i)}{ \sqrt{\sum_{x,y,z} (M(x,y,z)\cdot \mbox{FFT}^{-1}(\hat{T}_i\cdot \Omega_{ij} - T_i'))^2 }} \f]

    where \f[\hat{T_i} = \mbox{FFT}(Rot_{-\psi_i, -\phi_i, -\theta_i}(T_i(x,y,z))) \f]

    and \f$\Omega_{ij}\f$ is the overlap of the Fourier coefficients from
    spectrum after accounting for missing wedge, and rotations.

    The mean value \f$\bar{T}\f$ also accounts for the mask and the rotational
    constraint:

    \f[\bar{T} = \frac{1}{\sum_{x,y,z} M(x,y,z)} \sum_{x,y,z} \mbox{FFT}^{-1}(\hat{T_i}\cdot\Omega_{ij}) \f]
    
    The cross correlation remains defined as: 

    \f[cor(T_i, T_j) = \sum_{x,y,z} T_i'(x,y,z) \cdot T_j'(x,y,z) \f]
    
    @subsection FFT nad alignment search.

    We want to search for the best alignment of two volumes \f$V_1\f$, and \f$V_2\f$.

    The score of displacement (i,j,k) is:
    
    \f[
    S(i,j,k) = \sum_{x,y,z} V_1(x+i,y+j,z+k) \cdot V_2(x,y,z) 
    \f]

    This calculation is expensive.  Instead use FFT to do correlation.
    
    \f[
    S = \mbox{real}( \mbox{FFT}^{-1} ( \mbox{FFT}(V_1) \cdot \overline{\mbox{FFT}(V_2)} ) )
    \f]

    Here the product is element wise, and the bar signifies the conjugate
    should be used.

    From this correlation score we determine the displacement with the largest
    score, and return that as well as the score S(i,j,k).

*/

/** 
    Compute constrained correlation of two tomograms.  Search for best
    alignment between subtomograms.

    First we normalize each subtomogram taking into account the missing wedge
    and rotation.  The normalization is defined in Forster Eq #5.  This
    requires both masks since we need the intersection to calculate
    \f$\Omega\f$.  With both subtomograms normalized, we do a translational
    search to align them.

    @param vol1     first tomogram
    @param mask1    first mask in Fourier space.
    @param vol2     second tomogram
    @param mask2    second mask in Fourier space.
    @param ang1     Euler angle of how first volume is rotated.
    @param ang2     Euler angle of how second volume is rotated.
    
    @return Tuple containing the optimal offset to maximize correlation between
    subtomograms, and the constrained correlation score achieved at that
    displacement.

*/
std::tuple<arma::vec3, double> cons_corr_max(const arma::cube &v1, const arma::cube &m1, const arma::cube &v2, const arma::cube &m2, euler_angle ang);
//boost::tuple<arma::vec3, double> cons_corr_max(const arma::cube &v1, const arma::cube &m1, const arma::cube &v2, const arma::cube &m2, euler_angle ang);



/**
    @} // end group
*/

/**
    Determine the correlation between two volumes, using their representations in
    spherical harmonic coefficients at the given radius values.

    For each radius, sample the volume at that distance from the center over
    the surface of a sphere, and then calculate the spherical harmonic
    representation.  This gives a set of coefficients for every distance r.
    (\f$Y_l^m(r)\f$).  These are then used to calculate a correlation.

    Define \f[I = \int_r Y_{1l}^{m_1}(r) Y_{2l}^{*m_{2}}(r) r^2 dr \f]

    \f$Y_{1l}^{m_{1}}(r)\f$ is the \f$(l,m_1)\f$ coefficient from the first volume,
    from radius \f$r\f$. \f$Y_{2l}^{*m_{2}}(r)\f$ is the \f$(l,m_2)\f$ coefficient from
    the second volume conjugated.

    \f$I\f$ is a set of matrices for each \f$l\f$.  Each matrix of \f$I_l\f$
    has entries for each \f$(m_1,m_2)\f$.

    The value at \f$I_l(m_1,m_2)\f$ is found by integrating the product of the
    spherical harmonics and \f$r^2\f$ over the range of r for which both
    \f$m_1\f$ and \f$m_2\f$ are defined.

    Once \f$I\f$ is defined we sum over \f$l\f$ and multiply by Wigner matrices
    and take an inverse FFT to get the correlation.

    @param vol1     first volume
    @param vol2     second volume
    @param wig_d    Wigner small d matrices.
    @param radius   Radius values the coefficients are sampled from 
    @param max_l    the maximum degree of the spherical harmonic expansion
    @return correlation values

    @note this function has been combined with rot_search() from the MATLAB version.
*/
arma::cx_cube rot_search_cor(const arma::cube &vol1, const arma::cube &vol2, unsigned int max_l, const std::vector<double> &radius, const std::vector<arma::mat> &wig_d, arma::vec3 mid_co);


/**
    Generate a representation of the volume in spherical harmonics.

    Use the rot_search_expansion_single_shell to build representations of the
    volume, from its center with the given radii.  The list of spherical
    harmonics is representative of the entire volume.

    The coefficients are in a vector, one entry for each radius.  The entry
    for a given radius is the spherical harmonic coefficients for a surface
    with that radius.   The matrix has dimension (l+1)x(l+1).
    
    The coefficient for (l,m) is indexed at (l,m).
    
    The coefficient for a negative m, can be found from the positive entry,
    
    C(l,-m) = (-1)^m C^*(l,m), where C^(l,m) is the complex conjigate of the
    entry at l,m.
   
    This property holds as long as the data we take the transform of is real
    valued.

    @param vol the volume to represent
    @param max_l maximum degree of spherical harmonics to use
    @param radius the radii to use in sampling
    @return the set of spherical harmonic coefficients for each radius.
*/
std::vector<arma::cx_mat> rot_search_expansion(const arma::cube &vol, unsigned int max_l, const std::vector<double> &radius, arma::vec3 mid_co);


/**
    Expand a single shell into spherical coordinates.

    Sample points on the surface of a sphere of radius rad, centered at center,
    using values interpolated from the volume vol.  Then approximate the
    surface with spherical harmonics to generate a representation of the volume
    at this distance.

    @param ci  A cubic interpolater, allowing fast interpolation inside the volume. 
    @param center location of the center of the shell
    @param rad radius of the sampled shell
    @param max_l maximum degree of spherical harmonic to use in the approximation of the surface.
    @return the spherical harmonic coefficients for the given radius.  The matrix is lower triangular, and has entries for each (l,m) for m positive for which T Y_l^m coefficient is defined.
*/
arma::cx_mat rot_search_sh_expansion_single_shell(const cubic_interpolater &ci, arma::vec3 center, double rad, unsigned int max_l) ;


//std::vector<std::tuple<double, int, int, int> > local_max_index(const arma::cube, &score, unsigned int peak_spacing);

/**
   Find places which are local maximums in the cube.  The local maximum must be
   defined over a cubic mask region. 
   
   The mask is used to dilate the cube with periodic boundary conditions.  The
   local maximums are then extracted from the dilated cube, since these are the
   places where the dilated value is the same as the original.


   @param score Cube to dilate and extract maximum values from

   @return a list of coordinates into the cube which are maximal, and the
   scores that are associated with those locations.

*/
std::tuple<std::vector<euler_angle>, std::vector<double> > local_max_angles(const arma::cube &score, unsigned int peak_spacing);
//boost::tuple<std::vector<euler_angle>, std::vector<double> > local_max_angles(const arma::cube &score, unsigned int peak_spacing);


/**  
    Search for an optimal alignment of the two volumes.

    Search occurs in both rotational and translational space.  First do a rotational search, then for each candidate rotation do a search for optimal translational search.  Return all of the solutions found.

    @param vol1     a cubic volume of data. 
    @param mask1    a mask to be applied to the data.
    @param vol2     a cubic volume of data
    @param mask2    a mask to be applied to vol2
    @param max_l    maximum degree of spherical harmonic expansion to use.

    @returns A list of the best transformations found.  Each entry in the
    results list is a tuple containing the alignment correlation score, the
    translation, and the rotation.
*/
std::vector<std::tuple<double, arma::vec3, euler_angle> > combined_search( const arma::cube &vol1, const arma::cube &mask1, const arma::cube &vol2, const arma::cube &mask2, unsigned int max_l);
//std::vector<boost::tuple<double, arma::vec3, euler_angle> > combined_search( const arma::cube &vol1, const arma::cube &mask1, const arma::cube &vol2, const arma::cube &mask2, unsigned int max_l);


/**
  Compare two tuples based on their first item which is a double.

  This helper function is used to sort the combined search results.

  @param a  first tuple.
  @param b  second tuple
  @return true if a < b, false otherwise.

*/
bool tup_compare(std::tuple<double, arma::vec3, euler_angle> const &a, std::tuple<double, arma::vec3, euler_angle> const &b);
//bool tup_compare(boost::tuple<double, arma::vec3, euler_angle> const &a, boost::tuple<double, arma::vec3, euler_angle> const &b);

#endif

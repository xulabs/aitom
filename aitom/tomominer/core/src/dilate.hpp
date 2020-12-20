#ifndef TOMO_DILATE_HPP
#define TOMO_DILATE_HPP

#include <armadillo>

/**

    @defgroup dilation Dilation 
    @{

    Dilation of a 3d real valued volume.
    
    http://en.wikipedia.org/wiki/Dilation_%28morphology%29

    For a real valued function defined on the grid.  The dilation replaces each
    pixel with the maximum of the original image when the mask is ANDed with
    the image.


    For the square mask:

    \f[
    \begin{array}{ccc}
    0&0&0 \\
    1&1&1 \\
    0&1&0 
    \end{array}
    \f]

    Applying to the binary image:

    \f[
    \begin{array}{ccccccc}
    0&0&0&0&0&0&0 \\
    0&0&0&0&0&0&0 \\
    0&0&0&1&0&0&0 \\
    0&0&0&0&0&0&0 \\
    0&0&0&0&0&0&0 
    \end{array}
    \f]

    The dilation is:

    \f[
    \begin{array}{ccccccc}
    0& 0& 0 &0 &0 &0 &0 \\
    0& 0& 0 &1 &0 &0 &0 \\
    0& 0& 1 &1 &1 &0 &0 \\
    0& 0& 0 &0 &0 &0 &0 \\
    0& 0& 0 &0 &0 &0 &0
    \end{array}
    \f]

    The values which are one are those that have the 1 from the original in the
    mask.

    The algorithm we use is modified from Van Herk. @cite van_herk_1992  It has
    been modified to support periodic boundary conditions.
*/

/**

    Driver for Dilation.  Will call either dilate_small_se or dilate_large_se
    depending on size of structuring element.

    Dilate data in a cube according to a simple structuring element (cube of
    side length 2*l+1).
  
    This dilation works over periodic boundary conditions.  So in the
    event that the mask passes outside of the cube, it uses values from the
    other side of the volume.  
    
    This is done since the cubes we are working with correspond to angular
    space, and we will be using the dilation as a filter in finding the maximum
    values, so using periodic boundary conditions, is more reflective of the
    actual underlying structure of the angular coordinates.
    
    We use the algorithm described by Van Herk @cite van_herk_1992 to implement
    dilation.  The routine should fall back on a simpler method for small
    structuring elements.

    @param vol data to dilate. 
    @param se_width size of the structuring element.  SE will be a cube with side length se_width.
    @return the dilated cube.

    @todo support other shaped masks (structuring elements)
    @todo have periodic boundary support be a flag and support normal method as well.
*/
arma::cube dilate(const arma::cube &vol, int se_width);

/**
    Dilation using naive method.
    
    @param vol data to dilate. 
    @param se_width size of the structuring element.  SE will be a cube with side length se_width.
    @return the dilated cube.

*/
arma::cube dilate_small_se(const arma::cube &vol, int se_width);

/**
    Dilation using Van Herk algorithm.
    
    @param vol data to dilate. 
    @param se_width size of the structuring element.  SE will be a cube with side length se_width.
    @return the dilated cube.

*/
arma::cube dilate_large_se(const arma::cube &vol, int se_width);

/**
  @} // end group dilation
*/
#endif

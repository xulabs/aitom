#ifndef TOMO_AFFINE_TRANSFORM_HPP
#define TOMO_AFFINE_TRANSFORM_HPP

#include <armadillo>
#include "geometry.hpp"

/** 
    @addtogroup geometry
    @{

    @section aff_tran Affine Transforms

    @note This description does not apply to our code.  Our matrices are
    transposed compared to the description below.  The only real difference is
    that instead of right multiplying by column vectors, we left multiply by
    row vectors.  This does not matter much to users.

    An affine transformation matrix is an extension of rotation matrices with
    additional entries, which allow representation of translation, shear, and
    other deformations of space.

    The basic structure of an affine transformation matrix which represents a
    rotation and a displacement is:

    \f[
    A = 
    \left(
    \begin{array}{cc}
    R & \vec{b} \\
    0, \ldots, 0 & 1
    \end{array}
    \right)
    \f]

    where \f$R\f$ is a 3x3 rotation matrix and \f$\vec{b}\f$ is the 3x1
    translation component.

    These have several nice properties.  They can be used just as easily as the
    3x3 rotation matrices, and can be multiplied to generate compound
    transformations.  

    When this matrix is multiplied on the right by a column vector filled with
    a position, and a 1 in the fourth position, it will generate the new point
    after the transform.

    \f[
    \left(
    \begin{array}{c}
    \vec{y} \\
    1 
    \end{array}
    \right)
    =
    \left(
    \begin{array}{cc}
    R & \vec{b} \\
    0, 0, 0 & 1
    \end{array}
    \right)
    \left(
    \begin{array}{c}
    \vec{x} \\
    1
    \end{array}
    \right)
    \f]

    An affine matrix has the property that the inverse gives an inverse transformation.

    \f[
    A^{-1} = 
    \left(
    \begin{array}{cc}
    R^{-1} & -R^{-1}\vec{b} \\
    0, \ldots, 0 & 1
    \end{array}
    \right)
    \f]

    Using the orthonormality of \f$R (R^{-1} = R^{T}\f$:

    \f[
    A^{-1} = 
    \left(
    \begin{array}{cc}
    R^{T} & -R^{T}\vec{b} \\
    0, \ldots, 0 & 1
    \end{array}
    \right)
    \f]
    
    @section aff_tran_matlab Transformations in MATLAB

    In MATLAB the transform functions provided by the imaging toolbox use left
    multiplication by row vectors instead of right multiplication by column
    vectors.  Our code follows this standard.
*/

/**

    3D Affine transform object.  This can apply the transform or its inverse to vectors.

    @todo provide a multiplication method so tranform matrices can be combined naturally.
    @todo provide a constructor that is only displacement.
    @todo provide a constructor that is only a rotation.
*/
class affine_transform
{
    public:
        /**
            Construct a transform from a displacement and a rotation matrix. 
            @param rm rotation matrix to apply for the transform
            @param dx the displacement to include
        */    
        affine_transform(const rot_matrix &rm, const arma::vec3 &dx);
        
        /**
            Carry out a forward transform of the point x.
            @param x the point to transform
            @return the location of the transformed point.
            @see inverse()
        */
        arma::vec3 forward(const arma::vec3 &x);
        
        /**
            Carry out the reverse transform of the point x.
            @param x the point in transformed space
            @return the location of the point in original space.
            @see forward()
        */
        arma::vec3 inverse(const arma::vec3 &x);
        
        /** 
            Carry out an inverse transform using the transform matrix directly
            acting on a vector of length 4.
            
            This is the prefered method of using the transform since it
            involves no memory allocation.  If many transforms are needed it is
            better to build two length 4 vectors, and pass them in, instead of
            calling inverse() with the length 3 vectors, which have to be
            copied into length 4 vectors anyways.

            @param in vector to transform in homogenous coordinates ( [x,y,z,1.0] )
            @param out a vector to fill with the result of the transform in homogenous coordinates ([x',y',z',1.0]).
        
        */
        void inverse(const arma::vec4 &in, arma::vec4 &out);
        
        /** 
            Carry out a transform using the transform matrix directly
            acting on a vector of length 4.
            
            This is the prefered method of using the transform since it
            involves no memory allocation.  If many transforms are needed it is
            better to build two length 4 vectors, and pass them in, instead of
            calling inverse() with the length 3 vectors, which have to be
            copied into length 4 vectors anyways.

            @param in vector to transform in homogenous coordinates ( [x,y,z,1.0] )
            @param out a vector to fill with the result of the transform in homogenous coordinates ([x',y',z',1.0]).
        
        */
        void forward(const arma::vec4 &in, arma::vec4 &out);
        
        // operator*(affine_transform) 
        //affine_transform(const rot_matrix &rm);
        //affine_transform(const arma::vec3 &dx);

    private:
        /**
            This will modify the cached inverse transform matrix, reseting it to the inverse of the current forward transform. 
            This is called whenever the forward transform matrix is changed.
        */
        void update_inverse();
        /** The forward transform as a 4x4 matrix which can be left-multiplied by a row vector to generate a new position. */
        arma::mat44 A;
        /** The inverse of the matrix A.  Multiplying on the left by a row vector will give the point of origin for the given transformed point. */
        arma::mat44 Ai;
};

/** @} // geometry group add. */
#endif

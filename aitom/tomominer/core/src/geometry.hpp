#ifndef TOMO_GEOMETRY_HPP
#define TOMO_GEOMETRY_HPP

#include <armadillo>
#include <tuple>
//#include "boost/tuple/tuple.hpp"

class rot_matrix;
class euler_angle;


/** 
    @defgroup geometry Geometric objects and manipulations
    @{

    @section rot_mat Rotation matrices

    A rotation matrix(R) in 3D is a 3x3 orthogonal matrix, with determinant
    one, for which  \f$R^{-1} = R^T\f$.  

    All rotations described are right-handed.  For example, if we are at
    (0,0,>0) (hovering above the origin of the x,y plane, located along the
    positive z-axis looking back at the origin, a rotation around axis z by
    angle \f$\theta\f$ would be seen as a counter-clockwise rotation, following
    the right hand convention.  If you place your right hand such that the
    thumb points along the z-axis, the direction of positive rotation is given
    by the direction that your fingers wrap around the axis defined by the
    thumb.

    Rotation matrices can be combined by simple multiplication.  The three
    principle rotation matrices for three dimensions define rotation around
    each axis.  They are:

    Some References:

    http://en.wikipedia.org/wiki/Rotation_matrix

    http://www.sciencedirect.com/science/article/pii/S1047847705001231
    
    http://www.geometrictools.com/Documentation/EulerAngles.pdf


    \f[
    R_x(\theta) = \left(\begin{array}{ccc}
    1 & 0 & 0 \\
    0 & \cos \theta & -\sin \theta \\
    0 & \sin \theta  & \cos \theta
    \end{array}\right)
    \f]

    \f[
    R_y(\theta) = \left(\begin{array}{ccc}
    \cos \theta & 0 & \sin \theta \\
    0 & 1 & 0 \\
    -\sin \theta & 0 & \cos \theta
    \end{array}\right)
    \f]
    
    \f[
    R_z(\theta) = \left(\begin{array}{ccc}
    \cos \theta & -\sin \theta & 0 \\
    \sin \theta & \cos \theta & 0\\
    0 & 0 & 1
    \end{array}\right)
    \f]

    
    These combinations are useful for composing a matrix from Euler angles.  For
    example an Euler angle in ZYZ form of \f$(\theta,\phi,\psi)\f$ can be found in matrix form
    with \f$R_z(\theta)*R_y(\phi)*R_z(\psi)\f$.


    The matrices as written are compatible with multiplication by a column
    vector on the right side in order to generate the rotated position.  
    As an example, consider the application of the \f$R_x\f$ matrix to a position.
   

    \f[
    \left(
    \begin{array}{c}
    x' \\
    y' \\
    z'
    \end{array}
    \right)
    =
    \left(
    \begin{array}{ccc}
    1 & 0            &  0             \\
    0 & \cos \theta  & -\sin \theta   \\
    0 & \sin \theta  &  \cos \theta 
    \end{array}
    \right)
    \left(
    \begin{array}{c}
    x \\
    y \\
    z \\
    \end{array}
    \right)
    \f]

    @section euler_angles Euler Angles

    Euler angles provide a more compact representation then matrices for
    representing rotations.  

    We will use ZYZ right-handed Euler angles.  The letters give the axes each
    subsequent rotation is around, and the right-hand rule is the convention
    already described for defining the direction of rotation.

    Each subsequent rotation is about the newly defined axes, so we are able to
    write the entire transformation as a rotation matrix by simply multiplying
    the rotation matrices of the individual rotations.


    see: http://en.wikipedia.org/wiki/Euler_angles#Matrix_orientation

*/


/** 
    Return the center coordinate of a cube. 
    @param vol the cube.
    @return center coordinate for the cube.
*/
arma::vec3 get_center(const arma::cube &vol);

/**
    Return the center coordinate of a cube that has been fftshifted.  The center coordinate is often one-off from the actual center.
    @param vol The cube.
    @return center coordinate of the cube.
*/
arma::vec3 get_fftshift_center(const arma::cube &vol);

/**
    Filter angles based on distance.  If two angles are within cutoff, remove one from list.

    @param angs A vector of Euler zyz-right angles to convert.
    @param cutoff a distance to use for filtering.

    @todo this in O(N^2) in the number of angles.  Use a method to reduce this
    if there are often many angles that need to be considered.

*/
std::tuple<std::vector<euler_angle>, std::vector<double> > angle_list_redundancy_removal_zyz(std::vector<euler_angle> &angs, std::vector<double> &scores, double cutoff);
//boost::tuple<std::vector<euler_angle>, std::vector<double> > angle_list_redundancy_removal_zyz(std::vector<euler_angle> &angs, std::vector<double> &scores, double cutoff);


/**
    
    Return the angular distance between two Euler angles.

    The original version converts the angles to rotation matrices and then
    finds the L2 norm between those.  
    
    @todo Decide on a best distance method.

    Metrics for 3D Rotations: Comparison and Analysis
    Du Q. Huynh
    J Math Imaging Vis (2009) 35: 155–164
    DOI 10.1007/s10851-009-0161-2
   
    This is \Phi_5 from the paper, normalized to be between [0,1].
    
    For quaternions \f$q_1\f$ and \f$q_2\f$, the distance between them \f$\theta\f$ is:

    \f$\theta = cos^{-1}(2*dot(q1,q2)^2-1)\f$
    or
    \f$cos(theta) = 2 * dot(q1,q2)^2-1\f$;

    @param a1 ZYZ Euler angle
    @param a2 ZYZ Euler angle
    @return distance between two angles according to L2 norm of difference is distance matrices.
*/
double angle_difference_zyz(euler_angle &a1, euler_angle &a2);


/** 
    Representation of an Euler angle.

    A ZYZ right-hand rule Euler angle.
*/
class euler_angle : public arma::vec3 
{
public:

    euler_angle();
    euler_angle(const euler_angle &a);
    euler_angle(const arma::vec3 &v);
    euler_angle(double a0, double a1, double a2);
//    euler_angle(std::initializer_list<double> list);

    /**
        Convert the Euler angle to a rotation matrix.

        @return rotation matrix.
    */
    rot_matrix as_rot_matrix() const;

};


/** 
    rotation matrix for a given 3d rotation. 
*/
class rot_matrix: public arma::mat33 
{
    public:
        rot_matrix();
        rot_matrix(const arma::mat33 &m);
        rot_matrix(const rot_matrix &rm);
        rot_matrix(const euler_angle &ea);
        rot_matrix(const arma::vec3 &v);
        
        /** Construct from an initializer list.  Note because armadillo is column order, these need to be entered in that fashion.
        */
        //rot_matrix(std::initializer_list<double> list);

        /**
        convert rotation matrix to Euler angle.

        @return an Euler zyz-right handed rotation.
    */
    euler_angle as_euler_angle() const;
};
/**
  @} // end group geometry.
*/
#endif

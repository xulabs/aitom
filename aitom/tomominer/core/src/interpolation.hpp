#ifndef TOMO_INTERPOLATION_HPP
#define TOMO_INTERPOLATION_HPP

#include <armadillo>
/**

    @defgroup interpolation Interpolation methods
    @{

    All interpolation is done using convolution kernels as described by Keys @cite keys_1981.

    All interpolation methods are separable, so we can do each dimension independently.  

    For a function defined on grid points \f$x_k\f$ with values \f$c_k\f$ we
    define the value at a point \f$x\f$ as a convolution of the values with a
    kernel \f$u\f$.

    \f[
    g(x) = \sum_k c_k u(x-x_k)
    \f]

    The choice of kernel then determines the type of interpolation.  Using this
    kernel approach allows us to consider a wide variety of interpolations
    under the same formulation as a linear filter.


    @section nn_inter Nearest Neighbor Interpolation

    The easiest interpolation  method is nearest neighbor.  It can be thought
    of as a kernel:

    \f[
    u(x) = \left\{
            \begin{array}{lrcl}
            1 & \mbox{if } 0 <            &|x|& < \frac{1}{2} \\
            0 & \mbox{if } \frac{1}{2}\le &|x|&
            \end{array}
            \right.
    \f]


    This function when convoluted with a signal gives the nearest neighbor value for every point.


    @section linear_inter Linear Interpolation

    Linear interpolation can be written as a kernel:

    \f[
    u(x) = \left\{
           \begin{array}{lrcl}
            1-|x|   & \mbox{if } 0 <   &|x|& < 1 \\
            0       & \mbox{if } 1 \le &|x|&
           \end{array}
            \right.
    \f]

    @section cubic_inter Cubic Interpolation

    For cubic interpolation, there are several choices for free parameters
    which lead to different kernels.  Using the standard (\f$a =
    -\frac{1}{2}\f$) we have the standard kernel:
    
    \f[
    u(x) =  \left\{
            \begin{array}{lrcl}
            \frac{3}{2}|x|^3 - \frac{5}{2}|x|^2 + 1         & \mbox{if } 0 \le &|x|& < 1   \\
           -\frac{1}{2}|x|^3 +  \frac{5}{2}|x|^2 -4|x| + 2  & \mbox{if } 1 \le &|x|& < 2   \\
            0                                               & \mbox{if } 2 \le &|x|&       
            \end{array}
            \right.
    \f]

    Notice that the kernel requires points within the range (-2,2).  For points
    to be interpolated near the boundary, a value for the function at
    \f$c_{-1}\f$ and \f$c_{N+1}\f$ is needed for the interpolation to work.
    The values are from the Keys paper @cite keys_1981.  

    This kernel is identical to the Catmull-Rom spline.  The implementation as
    written uses a catmull_rom formulation for speed.

    The cubic interpolation function as as described by Keys is implemented
    here.  See the discussion at @ref vol_rot about differences with the MATLAB
    version when done in the context of transformations.

    @section multi_inter Multidimensional Interpolation

    For all interpolation methods here, we can carry out the interpolation in
    each dimension independently to perform 2,3, or higher dimensional
    interpolation.

*/
inline arma::vec4 catmull_rom_coeff(double x);

/**
    Base class for interpolation objects.

*/
class interpolater
{
    public:
        /**
            Initialize an interpolation object with a cube of data to be
            interpolated, and a value to return in case the requested point is
            outside of the volume and has to be extrapolated.

            @param f_ cube to interpolate.
            @param ext_val value to return in the event of extrapolation.
        */
        interpolater(const arma::cube &f_, double ext_val);

        /**
            Initialize an interpolation object with a cube of data.

            The ext_val is initialized to NaN.

            @param f_ data to be interpolated.
        */
        interpolater(const arma::cube &f_);

        //virtual ~interpolater();
        
        //void update_data(const arma::cube &f_);

        /**
            Return the interpolated value at position (x,y,z)

            If the position is outside of the cube, the value ext is returned instead.
            Otherwise the interpolation method chosen computes the best estimate for the value at the given position.
            
            @param x The x-coordinate to interpolate.
            @param y The y-coordinate to interpolate.
            @param z The z-coordinate to interpolate.

            @return estimation of f(x,y,z) where f is internal cube.
        */
        virtual double operator()(double x, double y, double z) const = 0;

        /**
            Evaluate the interpolation at the position (x(0), x(1), x(2)).

            @param x The coordinate to interpolate.
            @return estimation of f(x) where f is internal cube.
        
        */
        double operator()(const arma::vec &x) const;

        /** 
            Update the value returned when point requested is outside of cube.

            @param ext_val The new value.
        */
        void set_ext_val(double ext_val);

        /**
            Report the current out-of-bounds value.

            @return Value returned when coordinates outside of interpolation space.
        */
        double get_ext_val() const;

    protected:
        /**
            The cubic lattice interpolation is carried out on.
        */
        const arma::cube &data;
        /**
            The value returned if the coordinate requested to interpolate is outside of the cube.
        */
        double ext;
};
        

/**
    An interpolater that uses cubic splines to more accurately estimate the value at a given position.
*/
class cubic_interpolater : public interpolater
{
    public:
        cubic_interpolater(const arma::cube &f_, double ext_val);
        
        cubic_interpolater(const arma::cube &f_);

        virtual double operator()(double x, double y, double z) const;

        double operator()(const arma::vec &x) const;

    private:
        void update_data();
        arma::cube f;
};
        
/**
    An interpolater that uses the nearest values to interpolate at new points.
*/
class linear_interpolater : public interpolater
{
    public:
        linear_interpolater(const arma::cube &f_, double ext_val);

        linear_interpolater(const arma::cube &f_);
        
        double operator()(const arma::vec &x) const;

        virtual double operator()(double x, double y, double z) const;
};


class nearest_interpolater : public interpolater
{
    public:
        nearest_interpolater(const arma::cube &f_, double ext_val);
        
        nearest_interpolater(const arma::cube &f_);
        
        double operator()(double x, double y, double z) const;
};


/**
  @} // end group interpolation
*/
#endif

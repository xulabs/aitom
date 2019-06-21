#include "rotate.hpp"
#include "interpolation.hpp"

#include "arma_extend.hpp"

arma::cube transform(const interpolater &inter, affine_transform &at, arma::uvec3 size)
{
    arma::cube out(size(0), size(1), size(2));

    arma::vec4 x;
    arma::vec4 y;

    #pragma omp parallel for private(x,y)
    for(int x0 = 0; x0 < int(size(0)); x0++)
    {
        x(0) = x0;

        for(int x1 = 0; x1 < int(size(1)); x1++)
        {
            x(1) = x1;
            
            for(int x2 = 0; x2 < int(size(2)); x2++)
            {
                x(2) = x2;
                x(3) = 1.0;
                
                at.inverse(x,y);
                out(x(0),x(1),x(2)) = inter(y);
            }
        }
    }

    return out;
}

arma::cube rotate_vol(const arma::cube &vol, const rot_matrix &rm, const arma::vec3 &dx /* = {0,0,0} */)
{

    arma::vec3 center = get_center(vol);

    //arma::vec3 dx = -rm * center + center;
    arma::vec3 _dx = (-center.t() * rm + (center.t() + dx.t())).t();

    // our affine transformation matrix:
    affine_transform tform(rm, _dx);

    // the interpolation we will use.
    cubic_interpolater cub_int(vol, arma::math::nan());

    // do transformation.
    return transform(cub_int, tform, arma::shape(vol));
}

arma::cube rotate_vol_pad_mean(const arma::cube &vol, const rot_matrix &rm, const arma::vec3 &dx /* = {0,0,0} */)
{
    arma::cube vol2 = rotate_vol(vol, rm, dx);
    
    double s = 0;
    size_t n = 0;
    // calculate mean of locations that are finite, not NaN.
    for(size_t i = 0; i < vol2.n_elem; i++)
    {
        if(arma::is_finite(vol2(i)))
        {
            s+=vol2(i);
            n++;
        }
    }
    double mean = s / static_cast<double>(n);

    // fill in this average for all NaN/infinite values.
    for(size_t i = 0; i < vol2.n_elem; i++)
        if(!arma::is_finite(vol2(i)))
            vol2(i) = mean;

    return vol2;
}


arma::cube rotate_vol_pad_zero(const arma::cube &vol, const rot_matrix &rm, const arma::vec3 &dx /* = {0,0,0} */)
{
    arma::cube vol2 = rotate_vol(vol, rm, dx);

    for(size_t i = 0; i < vol2.n_elem; i++)
        if( !arma::is_finite(vol2(i)) )
            vol2(i) = 0;

    return vol2;
}


arma::cube rotate_mask(const arma::cube &mask, const rot_matrix &rm)
{
    // our output volume has the same dimensions as the input.
    arma::vec3 center = get_center(mask);

    //arma::vec3 dx = -rm * center + center;
    arma::vec3 dx = (-center.t() * rm + center.t()).t();

    // our affine transformation matrix:
    affine_transform tform(rm, dx);

    // the interpolation we will use.
    // linear is used to avoid negatives... but we still screen for them below?
    linear_interpolater lin_int(mask, 0);

    // do transformation.
    arma::cube mask_r = transform(lin_int, tform, arma::shape(mask));
    
    for(size_t i = 0; i < mask_r.n_elem; i++)
        if( mask_r(i) < 0 )
            mask_r(i) = 0.0;

    return mask_r;
}

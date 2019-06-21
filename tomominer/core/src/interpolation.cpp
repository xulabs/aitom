#include <vector>
#include <armadillo>

#include "arma_extend.hpp"
#include "interpolation.hpp"
#include "fatal_error.hpp"

using arma::span;

// attempt to gain a few seconds.  by doing this the sloppy way we avoid the
// nan/etc checks that make built-in floor so slow.
inline int _floor(double x){ return ((int)x); }

interpolater::interpolater(const arma::cube &f_) : data(f_), ext(arma::math::nan()) {}

interpolater::interpolater(const arma::cube &f_, double ext_val) : data(f_), ext(ext_val) {}

double interpolater::operator()(const arma::vec &x) const 
{ 
    if(x.n_elem < 3) 
        throw fatal_error() << "interpolater::operator() called with a vector of length < 3.";
    return (*this)(x(0), x(1), x(2)); 
}

void interpolater::set_ext_val(double ext_val) { ext = ext_val;}

double interpolater::get_ext_val() const { return ext; }


cubic_interpolater::cubic_interpolater(const arma::cube &f_) 
    : interpolater(f_)
{
    update_data();
}



cubic_interpolater::cubic_interpolater(const arma::cube &f_, double ext_val)
    : interpolater(f_, ext_val)
{
    update_data();
}

double cubic_interpolater::operator()(const arma::vec &x) const { return (*this)(x(0), x(1), x(2)); }

void cubic_interpolater::update_data()
{
    arma::uword M = data.n_rows;
    arma::uword N = data.n_cols;
    arma::uword P = data.n_slices;

    f = arma::cube(M+2, N+2, P+2);

    f(span(1,M), span(1,N), span(1,P)) = data;
    
    // fill in missing data:
    //
    //! @todo this is the slow but easy way to do this.  optimize.
    // 
    // We need three passes. In the first the sides are set, in the second the
    // edges, and in the third the corners.  In each additional round, the data
    // from before is still correct.  We do not need to actually calculate it
    // twice.  This was much less work however.  In the future, do this, then
    // iterate over edges, then corners to improve speed.
    for(arma::uword i = 0; i < 3; i++)
    {
        f(span((arma::uword)0),   span(), span()) = 3 * f(span((arma::uword)1), span(), span()) - 3 * f(span((arma::uword)2),   span(), span()) + f(span((arma::uword)3),   span(), span());
        f(span(M+1), span(), span()) = 3 * f(span(M), span(), span()) - 3 * f(span(M-1), span(), span()) + f(span(M-2), span(), span());

        f(span(), span((arma::uword)0),   span()) = 3 * f(span(), span((arma::uword)1), span()) - 3 * f(span(), span((arma::uword)2),   span()) + f(span(), span((arma::uword)3),   span());
        f(span(), span(N+1), span()) = 3 * f(span(), span(N), span()) - 3 * f(span(), span(N-1), span()) + f(span(), span(N-2), span());

        f(span(), span(), span((arma::uword)0)  ) = 3 * f(span(), span(), span((arma::uword)1)) - 3 * f(span(), span(), span((arma::uword)2)  ) + f(span(), span(), span((arma::uword)3)  );
        f(span(), span(), span(P+1)) = 3 * f(span(), span(), span(P)) - 3 * f(span(), span(), span(P-1)) + f(span(), span(), span(P-2));
    }
}
    

double cubic_interpolater::operator()(double x, double y, double z) const
{
    //Fudge factor to handle data that maps to slightly outside of the boundary.
    double EPSILON = 1e-13;

    if( x < 0 && x > -EPSILON) x = 0;
    if( y < 0 && y > -EPSILON) y = 0;
    if( z < 0 && z > -EPSILON) z = 0;
    if( x > data.n_rows   - 1 && x < data.n_rows   - 1 + EPSILON) x = data.n_rows   - 1;
    if( y > data.n_cols   - 1 && y < data.n_cols   - 1 + EPSILON) y = data.n_cols   - 1;
    if( z > data.n_slices - 1 && z < data.n_slices - 1 + EPSILON) z = data.n_slices - 1;

    if( x < 0 || data.n_rows-1 < x || y < 0 || data.n_cols-1 < y || z < 0 || data.n_slices-1 < z )
        return ext;

    unsigned int x0 = _floor(x);
    unsigned int y0 = _floor(y);
    unsigned int z0 = _floor(z);

    // allow the case where we are on the rightmost edge.
    if( x0 == data.n_rows-1   ) x0--;
    if( y0 == data.n_cols-1   ) y0--;
    if( z0 == data.n_slices-1 ) z0--;
    
    x -= x0;
    y -= y0;
    z -= z0;

    // remember we have padded the data!  x_{0} -> x{1}.  This is why indexing
    // starts at x0/y0/z0 instead of x0-1/y0-1/z0-1.
    
    arma::vec4 _x = catmull_rom_coeff(x);
    arma::vec4 _y = catmull_rom_coeff(y);
    arma::vec4 _z = catmull_rom_coeff(z);

    arma::vec4 yp,zp;

    for(size_t i = 0; i < 4; i++)
    {
        for(size_t j = 0; j < 4; j++)
            yp(j) = 0.5*arma::dot(f.slice(z0+i).col(y0+j).subvec(x0,x0+3), _x);
        zp(i) = 0.5*arma::dot(yp, _y);
    }
    return 0.5*arma::dot(zp, _z);
}


linear_interpolater::linear_interpolater(const arma::cube &f_, double ext_val) 
    : interpolater(f_, ext_val) {}

linear_interpolater::linear_interpolater(const arma::cube &f_) 
    : interpolater(f_) {}

double linear_interpolater::operator()(const arma::vec &x) const { return (*this)(x(0), x(1), x(2)); }

double linear_interpolater::operator()(double x, double y, double z) const
{
    //Fudge factor to handle data that maps to slightly outside of the boundary.
    double EPSILON = 1e-15;

    if( x < 0 && x > -EPSILON) x = 0;
    if( y < 0 && y > -EPSILON) y = 0;
    if( z < 0 && z > -EPSILON) z = 0;
    if( x > data.n_rows  - 1 && x < data.n_rows   - 1 + EPSILON) x = data.n_rows   - 1;
    if( y > data.n_cols  - 1 && y < data.n_cols   - 1 + EPSILON) y = data.n_cols   - 1;
    if( z > data.n_slices- 1 && z < data.n_slices - 1 + EPSILON) z = data.n_slices - 1;

    if( x < 0 || data.n_rows-1 < x || y < 0 || data.n_cols-1 < y || z < 0 || data.n_slices-1 < z )
        return ext;

    unsigned int x0 = _floor(x);
    unsigned int y0 = _floor(y);
    unsigned int z0 = _floor(z);

    // allow the case where we are on the rightmost edge.
    if( x0 == data.n_rows-1   ) x0--;
    if( y0 == data.n_cols-1   ) y0--;
    if( z0 == data.n_slices-1 ) z0--;
    
    int x1 = x0+1;
    int y1 = y0+1;
    int z1 = z0+1;

    x -= x0;
    y -= y0;
    z -= z0;

    double i1 = data(x0, y0, z0) * (1-z) + data(x0, y0, z1) * z;
    double i2 = data(x0, y1, z0) * (1-z) + data(x0, y1, z1) * z;
    double j1 = data(x1, y0, z0) * (1-z) + data(x1, y0, z1) * z;
    double j2 = data(x1, y1, z0) * (1-z) + data(x1, y1, z1) * z;

    double w1 = i1 * (1-y) + i2 * y;
    double w2 = j1 * (1-y) + j2 * y;

    return w1 * (1-x) + w2 * x;
}


inline arma::vec4 catmull_rom_coeff(double x)
{
    double x2 = x*x;
    arma::vec4 x_;
    x_(0) = x*((2-x)*x-1); 
    x_(1) = x2*(3*x-5)+2;
    x_(2) = x*((4-3*x)*x+1);
    x_(3) = x2*(x-1);
    return x_;
}

nearest_interpolater::nearest_interpolater(const arma::cube &f_, double ext_val) 
    : interpolater(f_, ext_val) {}

nearest_interpolater::nearest_interpolater(const arma::cube &f_) 
    : interpolater(f_) {}


double nearest_interpolater::operator()(double x, double y, double z) const
{
    if( x < 0 || data.n_rows-1 < x || y < 0 || data.n_cols-1 < y || z < 0 || data.n_slices-1 < z )
        return ext;

    unsigned int x0 = _floor(x);
    unsigned int y0 = _floor(y);
    unsigned int z0 = _floor(z);

    // allow the case where we are on the rightmost edge.
    if( x0 == data.n_rows-1   ) x0--;
    if( y0 == data.n_cols-1   ) y0--;
    if( z0 == data.n_slices-1 ) z0--;
    
    unsigned int x1 = x0+1;
    unsigned int y1 = y0+1;
    unsigned int z1 = z0+1;

    int xx = (x - x0 < x1 - x) ? x0 : x1;
    int yy = (y - y0 < y1 - y) ? y0 : y1;
    int zz = (z - z0 < z1 - z) ? z0 : z1;

    return data(xx,yy,zz);
}

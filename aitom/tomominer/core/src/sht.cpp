#include <iostream>
#include <cassert>

#include <armadillo>

#include "legendre.hpp"
#include "fft.hpp"
#include "fatal_error.hpp"


/* Solve a system using SVD based pseudo-inverse.  Use conditioning. */
arma::vec  solve_svd_pinv_cond(const arma::mat &A, const arma::vec &y)
{
    arma::mat U, V;
    arma::vec x;

    double cond = 1e-6;

    arma::svd_econ(U, x, V, A, 'b');

    // fix for conditioning.
    // this is what is different from arma::solve()
    for(size_t i = 0; i < x.n_elem; i++)
        if(x(i) < cond * max(x))
            x(i) = arma::datum::inf;

    arma::mat Gpi = V * arma::diagmat(1.0/x) * arma::trans(U);
    return Gpi * y;
}


arma::cx_mat forward_sht(arma::mat f, unsigned int L)
{
    // make sure data is periodic and delete last column.  This is ugly to make armadillo work.
    double diff = arma::sum( arma::abs( f(span(), 0) - f(span(), f.n_cols-1) ) );
    double tol  = arma::sum( arma::abs( f(span(), 0) - 0 )) * 1e-8;
    
    if( diff > tol )
    {
        //std::cerr << "first col:" << std::endl << f(span(), 0).t() << std::endl;
        //std::cerr << "last  col:" << std::endl << f(span(), f.n_cols-1).t() << std::endl;
        //std::cerr << "Error   = " << arma::sum( arma::abs( f(span(), 0) - f(span(), f.n_cols-1) ) ) << std::endl;
        //std::cerr << "forward_sht: data must be periodic.  first and last columns must be equal." << std::endl;

        //throw std::exception();
    }

    f = f(span(), span(0,f.n_cols-2));

    // make sure poles are constant
    if( arma::var(f(0, span())) > fabs(arma::mean(f(0,span()))) * 1e-8 )
    {
        //std::cerr << "forward_sht: var(f(0, span())) / fabs(arma::mean(f(0,span()))) = " << arma::var(f(0, span())) / fabs(arma::mean(f(0,span()))) << std::endl;
        //std::cerr << "forward_sht: data must include poles.  First row must be constant value." << std::endl;

        //throw std::exception();
    }
    if( arma::var(f(f.n_rows-1, span())) > fabs(arma::mean(f(f.n_rows-1,span()))) * 1e-8 )
    {
        //std::cerr << "forward_sht: arma::var(f(f.n_rows-1, span())) / fabs(arma::mean(f(f.n_rows-1,span()))) = " << arma::var(f(f.n_rows-1, span())) / fabs(arma::mean(f(f.n_rows-1,span()))) << std::endl;
        //std::cerr << "forward_sht: data must include poles.  Last row must be constant value." << std::endl;

        //throw std::exception();
    }

    unsigned int nlat = f.n_rows;
    unsigned int nlon = f.n_cols;

    // ugly int->double->int conversion...
    // Lnyq = Nyquist frequency.
    unsigned int Lnyq = std::min(ceil((nlon-1.0)/2.0), nlat-1.0);

    if( L > Lnyq || nlat < (L+1) )
    {
        std::cerr << "L value must satisfy Nyquist frequency requirements." << std::endl;
        throw std::exception();
    }

    // locations where we will evaluate P_j^m(x)
    arma::vec x = arma::cos(arma::linspace(0, M_PI, nlat));

    // We can generate P_j,m(cos theta_k) but with coefficients from Y_lm.
    
    // We need a semi-normalized legendre function, legendre.cpp provides fully
    // normalized, and unnormalized.  Here we do our own calculations and
    // coefficent corrections later.
    arma::cube P = generate_legendre(L, x);

    arma::cx_mat gfft(nlat, nlon);

    /**
    
     @todo Optimization: Swap row/col of f, since we operate on rows, so we can
     pass contiguous memory to fft().  Also fill columns of gfft() and then
     transpose later.
    
    @todo Optimization: Use a single plan since all of the rows are of the same
    size.  Maybe use the actual fftw interface here to speed things up.
    
    @todo Optimization: Consider the split imaginary representation for FFT,
    since we separate real/complex anyways in the next step, have fftw do that
    for us.
    
    @todo Optimization: make sure we are taking care of alignment by using
    fftw_malloc() to accelerate fft further.
    */
    
    for(size_t i = 0; i < gfft.n_rows; i++)
        gfft(i, span()) = fft(arma::conv_to<arma::rowvec>::from(f(i, span())));

    gfft *= 2.0*M_PI/nlon;

    for(size_t i = 0; i < gfft.n_rows; i++)
        for(size_t j = 0; j < gfft.n_cols; j++)
            if(abs(gfft(i,j)) < 1e-14)
                gfft(i,j) = 0;

    arma::cx_mat A_jm(L+1, L+1);
    A_jm.zeros();

    arma::mat a =  arma::real(gfft);
    arma::mat b = -arma::imag(gfft);

    a(span(), 0) = a(span(), 0) / 2;
    b(span(), 0) = b(span(), 0) / 2;

    for(size_t m = 0; m <= L; m++)
    {
        arma::mat pm(x.n_elem, L-m+1);

        for(size_t l = m, _l=0; l <= L; l++, _l++)
        {
            double sl = sqrt(2*l+1);
            double _q = 1.0;
            
            if( m > 0)
            {
                for(size_t q = l - m + 1; q <= l + m; q++)
                    _q *= q;
                _q /= 2.0;
            }

            for(size_t i = 0; i < x.n_elem; i++)
                // coefficents for semi-normalized. also includes sqrt(4*pi) term.
                pm(i,_l) = P(l, m, i) * sl / sqrt(_q) * M_PI;
        }

        if(m % 2 == 1)
            pm *= -1;
        
        // use SVD pseudo-inverse with conditioning correction.
        arma::vec coefr = solve_svd_pinv_cond(pm, a(span(), m));
        arma::vec coefi = solve_svd_pinv_cond(pm, b(span(), m));

        for(size_t i = 0; i < coefr.n_elem; i++)
        {
            A_jm(i+m, m) = std::complex<double>(coefr(i), -coefi(i));
            if(m % 2 == 1)
                A_jm(i+m, m) *= -1;
        }
    }

    A_jm *= sqrt(4*M_PI);

    return A_jm;
}


arma::mat reverse_sht(arma::cx_mat A_jm, unsigned int nlat, unsigned int nlon)
{
    arma::mat f = arma::zeros<arma::mat>(nlat, nlon);

    int L = A_jm.n_rows - 1;

    arma::vec theta = arma::linspace(0,   M_PI, nlat);
    arma::vec phi   = arma::linspace(0, 2*M_PI, nlon);

    arma::vec x = arma::cos(theta);
    
    arma::cube P = generate_normalized_legendre(L, x);

    arma::vec p1, p2;
    arma::mat Y = arma::zeros<arma::mat>(nlat, nlon);

    for(unsigned int l = 0; l < A_jm.n_rows; l++)
    {
        // m = 0 case.
        p1 = arma::ones<arma::vec>(phi.n_elem);

        arma::mat Y = arma::zeros<arma::mat>(nlat, nlon);
        for(size_t i = 0; i < nlat; i++)
            for(size_t j = 0; j < nlon; j++)
                Y(i,j) = P(l,0,i) * p1(j);

        f += std::real(A_jm(l,0)) * Y;
        f += std::imag(A_jm(l,0)) * Y;

        for(unsigned int m = 1; m <= l; m++)
        {
            p1 = sqrt(2) * cos(m*phi - M_PI/2.0);
            p2 = sqrt(2) * cos(m*phi);

            for(size_t i = 0; i < nlat; i++)
                for(size_t j = 0; j < nlon; j++)
                    Y(i,j) = P(l,m,i) * p1(j);

            f += std::imag(A_jm(l,m)) * Y;
            
            for(size_t i = 0; i < nlat; i++)
                for(size_t j = 0; j < nlon; j++)
                    Y(i,j) = P(l,m,i) * p2(j);

            f += std::real(A_jm(l,m)) * Y;
        }
    }
    return f;
}



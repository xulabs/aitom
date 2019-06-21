
#include <fftw3.h>
#include <armadillo>
/**
    @note For real valued inputs FFTW will fill only the first 1/2 of the
    array, the second half can be derived from the fact that the signal is
    hermetian for real valued inputs. For multidimensional input the array is
    filled as if the first dimension is halved. 
    
    @note FFTW will overwrite data while building plans!  Either call plan, and
    the fill input array, or use a planning method that is safe, FFTW_ESTIMATE
    for example.  

    @note For real valued inverse FFT, we assume that vector was generated from
    real data.  We use the Hermitian property of the vector \f$X\f$: (\f$X_k =
    X^*_{n-k}\f$), and use only the first \f$N/2+1\f$ coefficients and assume
    the rest are correct.
  
    @note All coefficients are returned. For real data \f$FFT(k_1,k_2) =
    FFT^*(n_1-k_1,n_2-k_2)\f$ and we must manually fill in the array from the
    first half.

*/

/***************************** 
  1D. 
*****************************/

arma::cx_vec fft(const arma::vec &X)
{
    arma::cx_vec out(X.n_elem);

    fftw_plan plan = fftw_plan_dft_r2c_1d(X.n_elem, (double *)(X.memptr()), (fftw_complex *)out.memptr(), FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    // fill in remainder for compatability with matlab.
    for(size_t i = out.n_elem/2+1; i < out.n_elem; i++)
        out(i) = conj(out(X.n_rows-i));

    return out;
}


arma::vec ifftr(const arma::cx_vec &X)
{
    arma::vec    out(X.n_elem);

    fftw_plan plan = fftw_plan_dft_c2r_1d(X.n_elem, (fftw_complex *)X.memptr(), out.memptr(), FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    out /= out.n_elem;
    return out;
}

arma::cx_vec fft(const arma::cx_vec &X)
{
    arma::cx_vec out(X.n_elem);

    fftw_plan plan = fftw_plan_dft_1d(X.n_elem, (fftw_complex *)X.memptr(), (fftw_complex *)out.memptr(), FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    return out;
}


arma::cx_vec ifft(const arma::cx_vec &X)
{
    arma::cx_vec out(X.n_elem);

    fftw_plan plan = fftw_plan_dft_1d(X.n_elem, (fftw_complex *)X.memptr(), (fftw_complex*)out.memptr(), FFTW_BACKWARD, FFTW_ESTIMATE);
    
    fftw_execute(plan);

    fftw_destroy_plan(plan);
    
    out /= out.n_elem;
    return out;
}



/***************************** 
  2D. 
*****************************/



arma::cx_mat fft(const arma::cx_mat &X)
{
    arma::cx_mat out(X.n_rows, X.n_cols);

    fftw_plan plan = fftw_plan_dft_2d(X.n_cols, X.n_rows, (fftw_complex *)X.memptr(), (fftw_complex *)out.memptr(), FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    return out;
}


arma::cx_mat ifft(const arma::cx_mat &X)
{
    arma::cx_mat out(X.n_rows, X.n_cols);

    fftw_plan plan = fftw_plan_dft_2d(X.n_cols, X.n_rows, (fftw_complex *)X.memptr(), (fftw_complex*)out.memptr(), FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    out /= out.n_elem;
    return out;
}

arma::cx_mat fft(const arma::mat &X)
{

    arma::cx_mat out = arma::zeros<arma::cx_mat>(X.n_rows / 2 + 1, X.n_cols);

    fftw_plan plan = fftw_plan_dft_r2c_2d(X.n_cols, X.n_rows, (double *)X.memptr(), (fftw_complex *)out.memptr(), FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    out.resize(X.n_rows, X.n_cols);

    // fill in missing data.
    for(size_t i = out.n_rows / 2 + 1; i < out.n_rows; i++)
    {
        out(i, 0) = conj(out(out.n_rows-i,0));

        for(size_t j = 1; j < out.n_cols; j++)
            out(i,j) = conj(out(out.n_rows-i, out.n_cols-j));
    }

    return out;
}

arma::mat ifftr(const arma::cx_mat &X)
{
    arma::cx_mat in = X(arma::span(0, X.n_rows / 2), arma::span());
    arma::mat    out(X.n_rows, X.n_cols);

    fftw_plan plan = fftw_plan_dft_c2r_2d(out.n_cols, out.n_rows, (fftw_complex *)in.memptr(), out.memptr(), FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    out /= out.n_elem;
    return out;
}



/***************************** 
  3D. 
*****************************/


arma::cx_cube fft(const arma::cube &X)
{
    arma::cx_cube out(X.n_rows / 2 + 1, X.n_cols, X.n_slices);

    fftw_plan plan = fftw_plan_dft_r2c_3d(X.n_slices, X.n_cols, X.n_rows, (double *)X.memptr(), (fftw_complex *)out.memptr(), FFTW_ESTIMATE);

    fftw_execute(plan);
    
    fftw_destroy_plan(plan);

    out.resize(X.n_rows, X.n_cols, X.n_slices);

    // fill X redundant data.
    for(size_t i = X.n_rows / 2 + 1; i < X.n_rows; i++)
    {
        out(i,0,0) = std::conj(out(X.n_rows-i, 0, 0));
        
        for(size_t j = 1; j < X.n_cols; j++)
            out(i,j,0) = std::conj(out(X.n_rows-i, X.n_cols-j, 0));
        
        for(size_t k = 1; k < X.n_slices; k++)
            out(i,0,k) = std::conj(out(X.n_rows-i, 0, X.n_slices-k));
        
        for(size_t j = 1; j < X.n_cols; j++)
            for(size_t k = 1; k < X.n_slices; k++)
                out(i,j,k) = std::conj(out(X.n_rows-i, X.n_cols-j, X.n_slices-k));
    }

    return out;
}

arma::cube ifftr(const arma::cx_cube &X)
{
    arma::cx_cube in = X(arma::span(0, X.n_rows / 2), arma::span(), arma::span());

    arma::cube ifft(X.n_rows, X.n_cols, X.n_slices);

    fftw_plan plan = fftw_plan_dft_c2r_3d(X.n_slices, X.n_cols, X.n_rows, (fftw_complex *)in.memptr(), ifft.memptr(), FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    return ifft/(X.n_rows * X.n_cols * X.n_slices);
}

arma::cx_cube fft(const arma::cx_cube &X)
{
    arma::cx_cube fft(X.n_rows, X.n_cols, X.n_slices);

    fftw_plan plan = fftw_plan_dft_3d(X.n_slices, X.n_cols, X.n_rows, (fftw_complex *)X.memptr(), (fftw_complex *)fft.memptr(), FFTW_FORWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    return fft;
}


arma::cx_cube ifft(const arma::cx_cube &X)
{
    arma::cx_cube ifft(X.n_rows, X.n_cols, X.n_slices);

    fftw_plan plan = fftw_plan_dft_3d(X.n_slices, X.n_cols, X.n_rows, (fftw_complex *)X.memptr(), (fftw_complex *)ifft.memptr(), FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan);

    fftw_destroy_plan(plan);

    return ifft/(X.n_rows * X.n_cols * X.n_slices);
}


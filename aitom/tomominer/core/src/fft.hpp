#ifndef TOMO_FFT_HPP
#define TOMO_FFT_HPP

#include <armadillo>

using arma::span;

/** 
    @defgroup fft FFT functions and utilities 
    @{

    Functions for computing the Fast Fourier Transform (FFT) and its inverse.
    These work over 1, 2, and 3 dimensional armadillo data types.  Both real
    and complex versions are supported.  In general the methods are designed to
    behave similar to Matlab/Octave in terms of the values they wil return.
    There are opportunities for efficiency gains in the real valued transforms
    in space and time since only half of the matrix is usually needed. 

    http://en.wikipedia.org/wiki/Fast_Fourier_transform

    http://mathworld.wolfram.com/FastFourierTransform.html

    @note We use the convention that the Forward FFT scales by 1, and the
    reverse scales by 1/N.  This is identical to MATLAB's behavior.
    
    @todo write in-place versions for complex->complex.
    
    @todo add fftshift functions for vec/mat.
*/


/** @name 1D Complex-Complex FFT functions. 
    @{
*/
/**
    1 dimensional FFT of complex data.
    
    @note there is no normalization applied for forward FFT.

    @param X data.
    @return FFT(X).
*/
arma::cx_vec  fft(const arma::cx_vec  &X);


/**
    1 dimensional inverse FFT of complex data.
    
    @note This is normalized by N = X.n_elem

    @param X FFT coefficients from fft.
    @return data from inverse FFT.
*/
arma::cx_vec  ifft(const arma::cx_vec  &X);

/**
    @}
*/

/** @name 1D Real-Complex FFT functions. 
    @{
*/
/**
    1 dimensional FFT of real data.
    
    @note there is no normalization applied for forward FFT.
    
    @note All coefficients are returned.  For real data \f$Y[k] = Y^*[n-k]\f$
    so not all coefficients need to be provided.
    
    @param X data.
    @return FFT(X).
*/
arma::cx_vec  fft(const arma::vec  &X);


/**
    1 dimensional inverse FFT of real data.
    
    @note This is normalized by N = X.n_elem

    @param X coefficients of real data.
    @return \f$FFT^{-1}(X)\f$.
*/
arma::vec     ifftr(const arma::cx_vec &X);


/**
    @}
*/

/** @name 2D Complex-Complex FFT functions. 
    @{
*/
/**
  2 dimensional FFT of complex data.

  @note there is no normalization in forward FFT.
  @param X the data to transform
  @return FFT(X)
*/
arma::cx_mat  fft(const arma::cx_mat  &X);


/**
  2 dimensional inverse FFT of complex data.

  @note there is normalization by X.n_rows * X.n_cols in the inverse FFT.
  @param X the coefficients to transform
  @return InverseFFT(X)
*/
arma::cx_mat  ifft(const arma::cx_mat  &X);


/**
    @}
*/

/** @name 2D Real-Complex FFT functions. 
    @{
*/
/**
  2 dimensional FFT of real data.

  @note there is no normalization in forward FFT.
  We fill in the missing data before returning.

  @param X the data to transform
  @return FFT(X)
*/
arma::cx_mat  fft(const arma::mat  &X);


/**
  2 dimensional inverse FFT of real data.

  @note there is normalization by X.n_rows * X.n_cols in the inverse FFT.
  @param X the coefficients to transform

  @return InverseFFT(X)
*/
arma::mat     ifftr(const arma::cx_mat &X);


/**
    @}
*/

/** @name 3D Complex-Complex FFT functions. 
    @{
*/
/**
  3 dimensional FFT of complex data.

  @note there is no normalization in forward FFT.
  @param X the data to transform
  @return FFT(X)
*/
arma::cx_cube fft(const arma::cx_cube &X);


/**
  3 dimensional inverse FFT of complex data.

  @note there is normalization by X.n_rows * X.n_cols * X.n_slices in the inverse FFT.
  @param X the coefficients to transform
  @return InverseFFT(X)
*/
arma::cx_cube ifft(const arma::cx_cube &X);


/**
    @}
*/

/** @name 3D Real-Complex FFT functions. 
    @{
*/
/**
  3 dimensional FFT of real data.

  @note there is no normalization in forward FFT.

  @param X the data to transform
  @return FFT(X)
*/
arma::cx_cube fft(const arma::cube &X);


/**
  3 dimensional inverse FFT of real data.

  @note there is normalization by X.n_rows * X.n_cols * X.n_slices in the inverse FFT.
  @param X the coefficients to transform
  @return InverseFFT(X)
*/
arma::cube    ifftr(const arma::cx_cube &X);



/**
    @}
*/
/** @name FFT utils 
    @{
*/

/** 
    Shift FFT output moving zero-frequency entry to center of array.  Shift is
    done in each dimension.

    @note fftshift is not its own inverse in the event that a dimension has an
    odd number of elements.  Always use ifftshift() to reverse a fftshift()
    call.
    
    @see ifftshift
    @param A cube of FFT coordinates.
    @return Shifted cube.
*/
template <class T>
arma::Cube<T> fftshift(const arma::Cube<T> &A)
{
    unsigned int p,n;
    
    arma::Cube<T> B(A), t(A);

    // for each dimension, swap.

    n = A.n_rows;
    p = (n + 1) / 2;
    

    t(span(0, n-p-1), span(), span()) = B(span(p,n-1), span(), span());
    t(span(n-p, n-1), span(), span()) = B(span(0,p-1), span(), span());

    B = t;

    n = A.n_cols;
    p = (n + 1) / 2;

    t(span(), span(0, n-p-1), span()) = B(span(), span(p,n-1), span());
    t(span(), span(n-p, n-1), span()) = B(span(), span(0,p-1), span());

    B = t;

    n = A.n_slices;
    p = (n + 1) / 2;
    t(span(), span(), span(0, n-p-1)) = B(span(), span(), span(p,n-1));
    t(span(), span(), span(n-p, n-1)) = B(span(), span(), span(0,p-1));

    return t;

}

/** 
    Undo a fftshift.  This moves the zero-frequency component back to first entry.

    @see fftshift
    @note fftshift is not its own inverse in the event that a dimension has an odd number of elements.  Always use ifftshift(fftshift()).
    @param A cube of FFT coordinates.
*/
template <class T>
arma::Cube<T> ifftshift(const arma::Cube<T> &A)
{
    unsigned int p,n;
    
    arma::Cube<T> B(A), t(A);

    // for each dimension, swap.

    n = A.n_rows;
    p = n - (n + 1) / 2;
    

    t(span(0, n-p-1), span(), span()) = B(span(p,n-1), span(), span());
    t(span(n-p, n-1), span(), span()) = B(span(0,p-1), span(), span());

    B = t;

    n = A.n_cols;
    p = n - (n + 1) / 2;

    t(span(), span(0, n-p-1), span()) = B(span(), span(p,n-1), span());
    t(span(), span(n-p, n-1), span()) = B(span(), span(0,p-1), span());

    B = t;

    n = A.n_slices;
    p = n - (n + 1) / 2;
    t(span(), span(), span(0, n-p-1)) = B(span(), span(), span(p,n-1));
    t(span(), span(), span(n-p, n-1)) = B(span(), span(), span(0,p-1));

    return t;
}


/**
    @}
*/
/**
  @} // end group fft.
*/
#endif

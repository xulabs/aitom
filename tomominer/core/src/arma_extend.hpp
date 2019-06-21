#ifndef TOMO_ARMA_EXTEND_HPP
#define TOMO_ARMA_EXTEND_HPP

#include <armadillo>

#include <tuple>
//#include "boost/tuple/tuple.hpp"

/**
    \file arma_extend.hpp

    @section Extensions to Armadillo.

    Extensions to the armadillo library.  These are mostly convenience
    functions that I have run across while trying to convert some MATLAB code.
    
    @todo integrate these better into the armadillo library.  Look at source for ideas on how to implement these functions.  Send anything good thats also useful back.
    @todo support for cube.is_vec(). cube.is_mat(), cube.is_*, and conversions between them.
    @todo add elementwise isnan/isfinite/etc... so we can do find(isnan(X)) and such.
    @todo figure out what arma_hot/arma_inline/etc... are, and how to use them to write armadillo functions.
*/


namespace arma {

/**
    Check if two Cubes are the same shape.

    This returns true if the two cubes have the same dimensions.  False otherwise.

    @param A first cube to compare
    @param B second cube to compare
    @return true if A and B are the same shape, false otherwise.
*/
template<class T,class U>
bool same_shape(const arma::Cube<T> &A, const arma::Cube<U> &B)
{
    return A.n_rows     == B.n_rows  \
        && A.n_cols     == B.n_cols  \
        && A.n_slices   == B.n_slices;
}

/**
    Check if two matrices are the same shape.

    This returns true if the two matrices have the same dimensions.  False
    otherwise.

    @param A first matrix to compare
    @param B second matrix to compare
    @return true if A and B are the same shape, false otherwise.

*/
template<class T,class U>
bool same_shape(const arma::Mat<T> &A, const arma::Mat<U> &B)
{
    return A.n_rows == B.n_rows \
        && A.n_cols == B.n_cols;
}

/*
template<typename eT>
arma_hot
arma_inline
void same_size(arma::vec<eT> &A, arma::vec<eT> &B)
{
    return (A.n_rows == B.n_rows && A.n_cols == B.n_cols);
}
*/

/* 
    Return the shape of the array in a vector.
*/

/**
  Determine the shape of the matrix.

  Store the shape into a vector.

  @param m the matrix to take the shape of.
  @return two element vector with number of rows and cols.
*/
template<class T>
arma::uvec2 shape(const arma::Mat<T> &m)
{
    arma::uvec2 s;
    s << m.n_rows << m.n_cols;
    return s;
}

/**
  Determine the shape of the cube.

  Store the shape into a vector.

  @param c the cube to take the shape of.
  @return three element vector with number of rows and cols.
*/
template<class T>
arma::uvec3 shape(const arma::Cube<T> &c)
{
    arma::uvec3 s;
    s << c.n_rows << c.n_cols << c.n_slices;
    return s;
}


/**

  Expand a cube into 3 cubes holding its indices in each dimension, where the indices are provided by v1, v2, and v3.

  This will return 3 volumes each the same size as the original.  For each position (i,j,k) in the original volume out1(i,j,k) = i; out2(i,j,k) = j; out3(i,j,k) = k

  @param v1 x coordinate for the cube
  @param v2 y coordinate for the cube
  @param v3 z coordinate for the cube

  @return three volumes with coordinates for the volume
*/
template<class T>
std::tuple<arma::Cube<T>, arma::Cube<T>, arma::Cube<T> > ndgrid(arma::Col<T> v1, arma::Col<T> v2, arma::Col<T> v3)
//boost::tuple<arma::Cube<T>, arma::Cube<T>, arma::Cube<T> > ndgrid(arma::Col<T> v1, arma::Col<T> v2, arma::Col<T> v3)
{
    arma::Cube<T> A1(v1.n_elem, v2.n_elem, v3.n_elem);
    arma::Cube<T> A2(v1.n_elem, v2.n_elem, v3.n_elem);
    arma::Cube<T> A3(v1.n_elem, v2.n_elem, v3.n_elem);

    //! @todo is there a more efficient way to do this?
    for(size_t i = 0; i < v1.n_elem; i++)
        for(size_t j = 0; j < v2.n_elem; j++)
            for(size_t k = 0; k < v3.n_elem; k++)
            {
                A1(i,j,k) = v1(i);
                A2(i,j,k) = v2(j);
                A3(i,j,k) = v3(k);
            }

    return std::make_tuple(A1, A2, A3);
    //return boost::make_tuple(A1, A2, A3);
}

/**
    Convert a integer offset into the volume into coordinates for accessing the
    cube according to its external access format.  This allows turning the
    results of find() back into cube/matrix coordinates.

    @param shape the dimensions of the volume to access.
    @param idx the offset into the cube.
    @return a coordinate which can be used to access the element.
*/
arma::uvec ind2sub(const arma::uvec &shape, int idx);
int sub2ind(const arma::uvec &shape, const arma::uvec &sub);


} // namespace arma


#endif

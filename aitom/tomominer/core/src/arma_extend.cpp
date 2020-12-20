#include <cstdlib> // div/ldiv

#include "arma_extend.hpp"

arma::uvec arma::ind2sub(const arma::uvec &shape, int idx)
{
    size_t d = shape.n_elem;

    arma::uvec sub(d);
    arma::uvec m(d);

    m(d-1) = 1;
    for(size_t i = 1; i < d; i++)
        m(d-i-1) = m(d-i)*shape(i-1);
    
    for(size_t i = 0; i < d; i++)
    {
        ldiv_t res = ldiv(idx, m(i));
        sub(d-i-1) = res.quot;
        idx        = res.rem;
    }

    return sub;
}

// according to the source code of arma::Cube::at() in Cube_meat.hpp
int arma::sub2ind(const arma::uvec &shape, const arma::uvec &sub) {
    size_t d = shape.n_elem;

    int idx = 0;
    int dim_element_num = 1;
    for(size_t d_i=0; d_i<d; d_i++) {
        idx += sub(d_i) * dim_element_num;
        dim_element_num *= shape(d_i);
    }

    return idx;
}



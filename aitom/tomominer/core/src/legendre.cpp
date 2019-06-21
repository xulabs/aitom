#include <iostream>

#include "legendre.hpp"

arma::cube generate_legendre(unsigned int l_max, const arma::vec &data)
{
    unsigned int i, m, l;
    unsigned int n = data.n_elem;

    arma::cube P = arma::zeros<arma::cube>(l_max+1, l_max+1, data.n_elem);

    // start with P(0,0) = 1.0;
    for(i = 0; i < n; i++)
        P(0, 0, i) = 1.0;

    arma::vec y(data.n_elem);
    for(i = 0; i < data.n_elem; i++)
        y(i) = sqrt(1.0 - data(i)*data(i));

    // using #2 build up diagonal.
    for(m = 1; m <= l_max; m++)
    {
        double c1 = (1 - 2.0*m);
        for(i = 0; i < n; i++)
            P(m,m,i) = c1 * y(i) * P(m-1,m-1,i);
    }
    
    // using #3 build off-diagonal entries.
    for(m=0; m < l_max; m++)
        for(i = 0; i < n; i++)
            P(m+1, m, i) = (2*m+1) * data(i) * P(m, m, i);

    // Now we can fill out remaining entries using relation #1.
    for(m =  0; m <= l_max; m++)
        for(l = m+2; l <= l_max; l++)
        {
            double c1 = (2.0*l-1.0) / (l-m);
            double c2 = (l+m-1.0) / (l-m);
            for(i = 0; i < n; i++)
                P(l, m, i) = c1 * data(i) * P(l-1, m, i) - c2 * P(l-2, m, i);
        }

    return P;
}

arma::cube generate_normalized_legendre(unsigned int l_max, const arma::vec &x)
{
    unsigned int i, m, l;
    unsigned int n = x.n_elem;

    arma::cube P(l_max+1, l_max+1, x.n_elem);

    P.zeros();

    // start with P(0,0) = 1.0;
    for(i = 0; i < n; i++)
        P(0, 0, i) = 1.0/sqrt(4.0*M_PI);
    
    // using #2 build up diagonal. sqrt(1-x^2) = sqrt((1-x)*(1+x))
    for(m = 1; m <= l_max; m++)
    {
        double c1 =  sqrt( ((double)(2*m+1)) / ((double)(2*m)) );
        for(i = 0; i < n; i++)
            P(m, m, i) = - c1 * sqrt(1-x(i)*x(i)) * P(m-1, m-1, i);
    }
    
    // using #3 build off-diagonal entries.
    for(m=0; m < l_max; m++)
    {
        double c1 = sqrt(2.0*m+3.0);
        for(i = 0; i < n; i++)
            P(m+1, m, i) = c1 * x(i) * P(m, m, i);
    }

    // Now we can fill out remaining entries using relation #1.
    for(m =  0; m <= l_max; m++)
        for(l = m+2; l <= l_max; l++)
        {
            double c1 = sqrt( ((2.0*l+1)*(2.0*l-1)) / ((l+m)*(l-m)));
            double c2 = sqrt( (2.0*l+1)*(l-m-1.0)*(l+m-1.0) / ((2.0*l-3)*(l-m)*(l+m)));
            for(i = 0; i < n; i++)
                P(l, m, i) = c1 * x(i) * P(l-1, m, i) - c2 * P(l-2, m, i);
        }

    return P;
}

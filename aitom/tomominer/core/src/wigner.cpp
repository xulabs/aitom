#include <vector>
#include <cassert>
#include <armadillo>

//#define parity(X) ((X % 2) == 0 ? 1 : -1)
/**
    Return parity or argument.

    @param x integer to test.
    @return 1 if X is even, -1 if X is odd.
*/
inline
int parity(int x)
{
    if(x % 2 == 0) return 1;
    return -1;
}



// The Wigner D-matrix of order l, is a matrix with entries from -l:l in both
// dimensions.  
std::vector<arma::mat> wigner_d(double theta, int L)
{
    int l, k, m;

    std::vector<arma::mat> D;
    
    // According to paper, recursion formulas are only valid for 0 < theta <= pi/2.0. 
    // Alternatives are given to adjust if we are out of range (Eqn #30).
    assert( 0 < theta && theta <= M_PI/2.0 );

    // generate matrices to be filled in.
    for(l = 0; l <= L; l++)
        D.push_back(arma::math::nan() * arma::ones<arma::mat>(2*l+1,2*l+1));

    // precompute sin/cos.
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);

    arma::mat g(L+1, L+1);
    g(0,0) = 1;
    
    // g recursion Eqn #29
    for(l = 1; l <= L; l++)
    {
        g(l,0) = sqrt( ((double)(2*l-1))/(2*l) ) * g(l-1,0);
        
        for(m = 1; m <= l; m++)
        {
            g(l,m) = sqrt( ((double)(l-m+1))/(l+m) )*g(l,m-1);
        }
    }
    
    // precompute sin/cos powers.
    arma::vec c1(L+1), c2(L+1);
    c1(0) = 1;
    c2(0) = 1;
    for(int i=1; i<=L;i++)
    {
        c1(i) = c1(i-1) * (1.0 + cos_theta);
        c2(i) = c2(i-1) * sin_theta;
    }


    // Fill in k=l. Eqn #28
    for(l=0; l <= L; l++)
        for(m=0; m <= l; m++)
            D[l](m+l,l+l) = parity(m+l) * g(l,m)* c1(m) * c2(l-m);
    
    // Precompute sin(theta) / (1+cos(theta)).
    double c3 = sin_theta / (1.0 + cos_theta);

    // Precompute 1.0/sqrt( (l*(l+1) - k*(k-1)) ).
    arma::mat C4(L+1, 2*L+1);

    for(l = 0; l <= L; l++)
        for(k = l; k > -l; k--)
            C4(l,k+L) = 1.0/sqrt( l*(l+1) - k*(k-1) );

    // Fill in bottom row: Eqn #26
    for(l=0; l<= L; l++)
        for(k = l; k > -l; k--)
            D[l](l+l,k-1+l) = (l+k) * C4(l,k+L) * c3 * D[l](l+l,k+l);

    // Fill in from bottom up. Eqn #25
    for(l = 0; l <= L; l++)
        for(m = l-1; m >= 0; m--)
            for(k = l; k > -l; k--)
                D[l](m+l,k-1+l) = sqrt(l*(l+1)-m*(m+1)) * C4(l,k+L) * D[l](m+1+l,k+l) + (m+k) * C4(l,k+L) * c3 * D[l](m+l,k+l);

    // fill in negative m. Eqn #27
    for(l = 0; l <= L; l++)
        for(m = -l; m < 0; m++)
            for(k = -l; k <= l; k++)
                D[l](m+l,k+l) = parity(m+k) * D[l](l-m,l-k);
    
    // Adjust the sign pattern.
    for(l = 0; l <= L; l++)
    {
        arma::vec alt(2*l+1);
        alt[0] = 1;
        for(m = 1; m < 2*l+1; m++)
            alt[m] = -alt[m-1];

        D[l] %= arma::toeplitz(alt, alt);
    }

    return D;
}


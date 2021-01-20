
#include <armadillo>

#include "dilate.hpp"
#include "arma_extend.hpp"
#include "fatal_error.hpp"

using arma::span;

//#define periodic(x,n) ((x < 0) ? n+x : ((x > n) ? x-n : x))
/**
    Apply perioidic boundary conditions to index i.  

    The index i may not lie in the range [0, n].  If it is outside the range,
    wrap it to be inside of the range.  Negative numbers wrap around from the
    right, n, n-1, n-2, etc. and numbers greater then n wrap around the left 0,
    1, 2, ...

    @param x the value to apply periodic boundary to.
    @param n the size of the dimension.
*/
inline
int periodic_bc(int x, int n)
{
    while(x < 0)
        x+=n;
    return (x % n);
}

arma::cube dilate(const arma::cube &vol, int se_width)
{
    if(se_width <= 2)
        throw fatal_error() << "dilate: structuring element size must be larger then 2.";
    if(se_width % 2 == 0)
        throw fatal_error() << "dilate: structuring element size must be odd";

    //! @todo find true break even point for when _large becomes faster.
    if( se_width <= 5 )
        return dilate_small_se(vol, se_width);
    else
        return dilate_large_se(vol, se_width);
}

arma::cube dilate_small_se(const arma::cube &vol, int se_width)
{
    int t = (se_width-1)/2;
    arma::cube out = -arma::math::inf() * arma::ones<arma::cube>(vol.n_rows, vol.n_cols, vol.n_slices);

    for(int i = 0; i < (int)vol.n_rows; i++)
    {
        for(int j = 0; j < (int)vol.n_cols; j++)
        {
            for(int k = 0; k < (int)vol.n_slices; k++)
            {
                // new value at (i,j,k) will be max of values within mask & cube ceneterd at (i,j,k).
                for(int ii = -t; ii <= t; ii++)
                {
                    int i_ = periodic_bc(i+ii, (int)vol.n_rows);

                    for(int jj = -t; jj <= t; jj++)
                    {
                        int j_ = periodic_bc(j+jj, (int)vol.n_cols);

                        for(int kk = -t; kk <= t; kk++)
                        {
                            int k_ = periodic_bc(k+kk, (int)vol.n_slices);

                            out(i,j,k) = std::max(out(i,j,k), vol(i_, j_, k_));
                        }
                    }
                }
            }
        }
    }

    return out;
}


arma::cube dilate_large_se(const arma::cube &vol, int se_width)
{

    int t = (se_width-1)/2;

    arma::uvec3 N = arma::shape(vol);
    arma::ivec3 pad;

    for(size_t i = 0; i < 3; i++)
        pad(i) = (N(i) % se_width == 0) ? 0 : se_width - (N(i) % se_width);

    arma::ivec3 M = N + pad + 2*t;

    arma::cube r(M(0), M(1), M(2));
    
    arma::vec g;
    arma::vec h;

    r(span(t,N(0)-1+t), span(t,N(1)-1+t), span(t,N(2)-1+t)) = vol;

    // padding.
    r(span(),                   span(),                 span(0,      t-1)   ) = r(span(),                     span(),                     span(N(2), N(2)+t-1    )); 
    r(span(),                   span(),                 span(N(2)+t, M(2)-1)) = r(span(),                     span(),                     span(t,    2*t+pad(2)-1));

    r(span(),                   span(0,      t-1   ), span()                ) = r(span(),                     span(N(1), N(1)+t-1    ), span()                    );
    r(span(),                   span(N(1)+t, M(1)-1), span()                ) = r(span(),                     span(t,    2*t+pad(1)-1), span()                    );

    r(span(0,      t-1   ),   span(),                 span()                ) = r(span(N(0), N(0)+t-1    ), span(),                     span()                    );
    r(span(N(0)+t, M(0)-1),   span(),                 span()                ) = r(span(t,    2*t+pad(0)-1), span(),                     span()                    );

    g.set_size(M(2));
    h.set_size(M(2));

    for(int i = t; i < M(0)-t; i++)
    {
        for(int j = t; j < M(1)-t; j++)
        {
            // build g()
            g(0) = r(i,j,0);
            for(int x = 1; x < M(2); x++)
            {
                if( x % se_width == 0)
                    g(x) = r(i,j,x);
                else
                    g(x) = std::max(g(x-1), r(i,j,x));
            }

            // build h()
            h(M(2)-1) = r(i,j,M(2)-1);
            for(int x = M(2)-2; x >= 0; x--)
            {
                if( x % se_width == 0)
                    h(x) = r(i,j,x);
                else
                    h(x) = std::max(h(x+1), r(i,j,x));
            }

            // merge.
            for(int x = t; x < M(2)-t; x++)
                r(i,j,x) = std::max(g(x+t),h(x-t));
        }
    }

    // padding.
    r(span(),                   span(),                 span(0,      t-1)   ) = r(span(),                     span(),                     span(N(2), N(2)+t-1    )); 
    r(span(),                   span(),                 span(N(2)+t, M(2)-1)) = r(span(),                     span(),                     span(t,    2*t+pad(2)-1));

    r(span(),                   span(0,      t-1   ), span()                ) = r(span(),                     span(N(1), N(1)+t-1    ), span()                    );
    r(span(),                   span(N(1)+t, M(1)-1), span()                ) = r(span(),                     span(t,    2*t+pad(1)-1), span()                    );

    r(span(0,      t-1   ),   span(),                 span()                ) = r(span(N(0), N(0)+t-1    ), span(),                     span()                    );
    r(span(N(0)+t, M(0)-1),   span(),                 span()                ) = r(span(t,    2*t+pad(0)-1), span(),                     span()                    );

    g.set_size(M(1));
    h.set_size(M(1));

    for(int i = t; i < M(0)-t; i++)
    {
        for(int j = t; j < M(2)-t; j++)
        {
            // build g()
            g(0) = r(i,0,j);
            for(int x = 1; x < M(1); x++)
            {
                if( x % se_width == 0)
                    g(x) = r(i,x,j);
                else
                    g(x) = std::max(g(x-1), r(i,x,j));
            }

            // build h()
            h(M(1)-1) = r(i,M(1)-1,j);
            for(int x = M(1)-2; x >= 0; x--)
            {
                if( x % se_width == 0)
                    h(x) = r(i,x,j);
                else
                    h(x) = std::max(h(x+1), r(i,x,j));
            }

            // merge.
            for(int x = t; x < M(1)-t; x++)
                r(i,x,j) = std::max(g(x+t),h(x-t));
        }
    }

    // padding.
    r(span(),                   span(),                 span(0,      t-1)   ) = r(span(),                     span(),                     span(N(2), N(2)+t-1    )); 
    r(span(),                   span(),                 span(N(2)+t, M(2)-1)) = r(span(),                     span(),                     span(t,    2*t+pad(2)-1));

    r(span(),                   span(0,      t-1   ), span()                ) = r(span(),                     span(N(1), N(1)+t-1    ), span()                    );
    r(span(),                   span(N(1)+t, M(1)-1), span()                ) = r(span(),                     span(t,    2*t+pad(1)-1), span()                    );

    r(span(0,      t-1   ),   span(),                 span()                ) = r(span(N(0), N(0)+t-1    ), span(),                     span()                    );
    r(span(N(0)+t, M(0)-1),   span(),                 span()                ) = r(span(t,    2*t+pad(0)-1), span(),                     span()                    );

    g.set_size(M(0));
    h.set_size(M(0));
    for(int i = t; i < M(1); i++)
    {
        for(int j = t; j < M(2); j++)
        {
            // build g()
            g(0) = r(0,i,j);
            for(int x = 1; x < M(0); x++)
            {
                if( x % se_width == 0)
                    g(x) = r(x,i,j);
                else
                    g(x) = std::max(g(x-1), r(x,i,j));
            }

            // build h()
            h(M(0)-1) = r(M(0)-1,i,j);
            for(int x = M(0)-2; x >= 0; x--)
            {
                if( x % se_width == 0)
                    h(x) = r(x,i,j);
                else
                    h(x) = std::max(h(x+1), r(x,i,j));
            }

            // merge.
            for(int x = t; x < M(0)-t; x++)
                r(x,i,j) = std::max(g(x+t),h(x-t));
        }
    }

    return r(span(t,N(0)+t-1), span(t,N(1)+t-1), span(t,N(2)+t-1));

}

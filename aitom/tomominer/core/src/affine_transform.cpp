#include "affine_transform.hpp"

using arma::span;

affine_transform::affine_transform(const rot_matrix &rm, const arma::vec3 &dx)
{

    A(span(0,2), span(0,2)) = rm;
    A(span(0,2), 3        ) = arma::zeros<arma::mat>(3,1);
    A(3,         span(0,2)) = dx.t();
    A(3,         3        ) = 1;

    update_inverse();
}

void affine_transform::update_inverse()
{
    Ai = A.t();

    Ai(span(0,2),   3        ) = arma::zeros<arma::mat>(3,1);
    Ai(3,           span(0,2)) = -A(3,span(0,2)) * Ai(span(0,2), span(0,2));
    Ai(3,           3        ) = 1;
}

arma::vec3 affine_transform::forward(const arma::vec3 &x)
{
    arma::rowvec4 xp;
    xp << x(0) << x(1) << x(2) << 1.0;
    arma::rowvec4 u = xp * A;
    return u.subvec(0,2).t();
}


arma::vec3 affine_transform::inverse(const arma::vec3 &x)
{
    arma::rowvec4 xp;
    xp << x(0) << x(1) << x(2) << 1.0;
    arma::rowvec4 u = xp * Ai;
    return u.subvec(0,2).t();
}

void affine_transform::inverse(const arma::vec4 &in, arma::vec4 &out)
{
    out = (in.t() * Ai).t();
}

void affine_transform::forward(const arma::vec4 &in, arma::vec4 &out)
{
    out = (in.t() * A).t();
}

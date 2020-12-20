
#include <iostream> // warnings about non-unique euler angles.

#include "geometry.hpp"

euler_angle::euler_angle()
{
    (*this)(0) = 0;
    (*this)(1) = 0;
    (*this)(2) = 0;
}

euler_angle::euler_angle(const euler_angle &a)
{
    (*this)(0) = a(0);
    (*this)(1) = a(1);
    (*this)(2) = a(2);
}

euler_angle::euler_angle(const arma::vec3 &v)
{
    (*this)(0) = v(0);
    (*this)(1) = v(1);
    (*this)(2) = v(2);
}

/*
euler_angle::euler_angle(std::initializer_list<double> list)
{
    if(list.size() != 3)
        throw std::exception();
    size_t i = 0;
    for(auto it = list.begin(); it != list.end(); it++)
        (*this)(i++) = *it;
}
*/

euler_angle::euler_angle(double a0, double a1, double a2)
{
    (*this)(0) = a0;
    (*this)(1) = a1;
    (*this)(2) = a2;
}


rot_matrix::rot_matrix()
{
    for(size_t i = 0; i < 3; i++)
        for(size_t j = 0; j < 3; j++)
            (*this)(i,j) = (i==j) ? 1 : 0;
}

rot_matrix::rot_matrix(const rot_matrix &rm)
{
    for(size_t i = 0; i < rm.n_elem; i++)
        (*this)(i) = rm(i);
}

rot_matrix::rot_matrix(const arma::mat33 &m)
{
    for(size_t i = 0; i < m.n_elem; i++)
        (*this)(i) = m(i);
}

rot_matrix::rot_matrix(const euler_angle &ea)
{
    (*this) = ea.as_rot_matrix();
}

rot_matrix::rot_matrix(const arma::vec3 &v)
{
    (*this) = euler_angle(v).as_rot_matrix();
}

/*
rot_matrix::rot_matrix(std::initializer_list<double> list)
{
    if(list.size() != 9)
        throw std::exception();
    size_t i = 0;
    for(auto it = list.begin(); it != list.end(); it++)
        (*this)(i++) = *it;
}
*/

rot_matrix euler_angle::as_rot_matrix() const
{
    double s1 = sin((*this)(0)), c1 = cos((*this)(0));
    double s2 = sin((*this)(1)), c2 = cos((*this)(1));
    double s3 = sin((*this)(2)), c3 = cos((*this)(2));

    // Filling with initializer list fills column order in armadillo.
    /*
    return rot_matrix({
              c1*c2*c3 - s1*s3   , -c3*s1 - c1*c2*s3,   c1*s2,
              c1*s3    + c2*c3*s1,  c1*c3 - c2*s1*s3,   s1*s2,
             -c3*s2              ,  s2*s3,              c2    });
    */
    rot_matrix rm;
    rm(0,0) = c1*c2*c3 - s1*s3;
    rm(1,0) = -c3*s1 - c1*c2*s3;
    rm(2,0) = c1*s2;
    rm(0,1) = c1*s3 + c2*c3*s1;
    rm(1,1) = c1*c3 - c2*s1*s3;
    rm(2,1) = s1*s2;
    rm(0,2) = -c3*s2;
    rm(1,2) = s2*s3;
    rm(2,2) = c2;
    return rm;
}


/*
    Using method found in: http://www.geometrictools.com/Documentation/EulerAngles.pdf
    NOTE: our matrices are transposed from those described there.
*/
euler_angle rot_matrix::as_euler_angle() const
{
    euler_angle ea;

    if( (*this)(2,2) < 1 )
    {
        if( (*this)(2,2) > -1 )
        {
            ea(0) =  atan2( (*this)(2,1),  (*this)(2,0) );
            ea(1) =  acos((*this)(2,2));
            ea(2) =  atan2( (*this)(1,2), -(*this)(0,2) );
        }
        else // (*this)(2,2) == -1
        {
            // solution is not unique: ea(2)-ea(0) = atan2( (*this)(1,0), (*this)(1,1) )
            //std::cerr << "matrix.as_euler_angle(): euler angle definition is not unique. (ea(2) - ea(0) == atan(r(1,0), r(1,1))).  Using convention that ea(2) = 0." << std::endl;
            ea(0) = -atan2( (*this)(0,1),  (*this)(1,1) );
            ea(1) =  M_PI;
            ea(2) =  0;
        }
    }
    else // (*this)(2,2) == 1
    {
        // solution is not unique: ea(2)+ea(0) = atan2( (*this)(1,0), (*this)(1,1) )
        //std::cerr << "matrix.as_euler_angle(): euler angle definition is not unique. (ea(2) + ea(0) == atan(r(1,0), r(1,1))).  Using convention that ea(2) = 0." << std::endl;
        ea(0) = atan2((*this)(0,1), (*this)(1,1));
        ea(1) = 0;
        ea(2) = 0;
    }
    return ea;
}

arma::vec3 get_center(const arma::cube &vol)
{
    arma::vec3 x;
    x(0) = ceil(vol.n_rows  /2.0);
    x(1) = ceil(vol.n_cols  /2.0);
    x(2) = ceil(vol.n_slices/2.0);
    return x;
}

arma::vec3 get_fftshift_center(const arma::cube &vol)
{
    arma::vec3 x;
    x(0) = floor(vol.n_rows  /2.0);
    x(1) = floor(vol.n_cols  /2.0);
    x(2) = floor(vol.n_slices/2.0);
    return x;
}


std::tuple<std::vector<euler_angle>, std::vector<double> > angle_list_redundancy_removal_zyz(std::vector<euler_angle> &angs, std::vector<double> &scores, double cutoff)
//boost::tuple<std::vector<euler_angle>, std::vector<double> > angle_list_redundancy_removal_zyz(std::vector<euler_angle> &angs, std::vector<double> &scores, double cutoff)
{
    std::vector<euler_angle> keep_ang;
    std::vector<double> keep_scr;

    std::vector<bool> redundant(angs.size());

    for(unsigned int i = 0; i < angs.size(); i++)
        redundant[i] = false;

    for(unsigned int i = 0; i < angs.size(); i++)
    {
        if(redundant[i])
            continue;
        for(unsigned int j = i+1; j < angs.size(); j++)
        {
            if(redundant[j])
                continue;
            redundant[j] = (angle_difference_zyz(angs[i], angs[j]) < cutoff);
        }
    }

    for(unsigned int i = 0; i < redundant.size(); i++)
        if(!redundant[i])
        {
            keep_ang.push_back(angs[i]);
            keep_scr.push_back(scores[i]);
        }
    return std::make_tuple(keep_ang, keep_scr);
    //return boost::make_tuple(keep_ang, keep_scr);
}


double angle_difference_zyz(euler_angle &a1, euler_angle &a2)
{
    rot_matrix m1 = a1.as_rot_matrix();
    rot_matrix m2 = a2.as_rot_matrix();
    return arma::norm( arma::eye<arma::mat>(3,3) - m1 * m2.t(), "fro");// / (2.0*sqrt(2.0));
}

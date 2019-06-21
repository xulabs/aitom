
#include <vector>

#include <tuple>
//#include "boost/tuple/tuple.hpp"

#include <armadillo>

#include "align.hpp"

using arma::span;

std::complex<double> trapz(const arma::vec &x, const arma::cx_vec &y)
{
    if(x.n_elem != y.n_elem) 
        throw fatal_error() << "trapz: x and y must be same lengths. len(x) = " << x.size() << " len(y) = " << y.size();

    if(x.n_elem == 1)
        return y(0);

    std::complex<double> sum = 0.0;
    for(size_t i = 1; i < x.n_elem; i++)
        sum += (x[i] - x[i-1]) * (y[i] + y[i-1]);

    return 0.5 * sum;
}


std::tuple<arma::vec3, double> cons_corr_max(const arma::cube &vol1, const arma::cube &mask1, const arma::cube &vol2, const arma::cube &mask2, euler_angle ang)
//boost::tuple<arma::vec3, double> cons_corr_max(const arma::cube &vol1, const arma::cube &mask1, const arma::cube &vol2, const arma::cube &mask2, euler_angle ang)
{
    rot_matrix rm = ang.as_rot_matrix();

    arma::cube v2 = rotate_vol_pad_mean(vol2, rm);
    arma::cube m2 = rotate_mask(mask2, rm);

    arma::cube mask = mask1 % m2;
    
    arma::cx_cube vol1_fft = fft(vol1);
    arma::cx_cube vol2_fft = fft(v2);
    
    vol1_fft(0,0,0) = 0;
    vol2_fft(0,0,0) = 0;
    
    vol1_fft = fftshift(vol1_fft) % mask;
    vol2_fft = fftshift(vol2_fft) % mask;

    vol1_fft /= sqrt(arma::accu(arma::square(arma::abs(vol1_fft))));
    vol2_fft /= sqrt(arma::accu(arma::square(arma::abs(vol2_fft))));

    arma::cx_cube tmp = vol1_fft % arma::conj(vol2_fft);

    arma::cube corr = arma::real(fft(ifftshift( tmp )));

    // search for maxima values in the correlation.
    arma::uvec3 max_loc;
    double max_val = corr.max(max_loc(0), max_loc(1), max_loc(2));

    arma::vec3 pos;
    arma::uvec3 siz = arma::shape(corr);
    for(size_t i = 0; i < 3; i++)
    {
        pos(i) = max_loc(i);
        if(pos(i) > siz(i)/2)
            pos(i) -= siz(i);
    }
    pos = -pos;
    
    return std::make_tuple(pos, max_val);
    //return boost::make_tuple(pos, max_val);
}

arma::cx_cube rot_search_cor(const arma::cube &vol1, const arma::cube &vol2, unsigned int L, const std::vector<double> &radius, const std::vector<arma::mat> &wig_d, arma::vec3 mid_co)
{
    // first generate representation of each volume as a set of spherical
    // harmonic coefficients at different radii from the center.
    std::vector<arma::cx_mat> coef1 = rot_search_expansion(vol1, L, radius, mid_co);
    std::vector<arma::cx_mat> coef2 = rot_search_expansion(vol2, L, radius, mid_co);
    
    arma::cx_cube I = arma::zeros<arma::cx_cube>(L+1, 2*L+1, 2*L+1);

    arma::cx_cube It_old;

    for(size_t rad_i = 0; rad_i < radius.size(); rad_i++)
    {
        arma::cx_cube It = arma::zeros<arma::cx_cube>(L+1, 2*L+1, 2*L+1);
        double r2 = radius[rad_i]*radius[rad_i];

        size_t num_entries = std::min(coef1[rad_i].n_rows, coef2[rad_i].n_rows);

        arma::cx_vec c1(2*L+1), c2(2*L+1);

        //for l = 0 : L (here we use num_entries because our matrix may not be filled out to LxL and we need those numbers to be zero.)
        for(arma::uword l = 0; l < num_entries; l++)
        {
            c1.zeros();
            c2.zeros();

            c1(L) = coef1[rad_i](l, 0);
            c2(L) = coef2[rad_i](l, 0);

            for(size_t m = 1; m <= l; m++)
            {
                std::complex<double> c;
               
                c = coef1[rad_i](l, m);

                c1(L-m) = 1.0/sqrt(2.0) * conj(c);
                c1(L+m) = 1.0/sqrt(2.0) * c       * ((m % 2 == 1) ? -1.0 : 1.0);
                
                c = coef2[rad_i](l, m);
                
                c2(L-m) = 1.0/sqrt(2.0) * conj(c);
                c2(L+m) = 1.0/sqrt(2.0) * c       * ((m % 2 == 1) ? -1.0 : 1.0);
                
            }

            It(span(l), span(), span()) = c1 * c2.t() * r2;
        }

        if(radius.size() == 1)
        {
            I = It;
            break;
        }

        if(rad_i > 0)
        {
            arma::cx_cube temp = It_old + It;
            I += (radius[rad_i] - radius[rad_i-1]) * (It_old + It);
        }

        It_old = It;

    }

    I /= 2.0;

    // angular spacing. 2*pi/(2*L). [0, 2*pi].
    arma::cx_cube TF = arma::zeros<arma::cx_cube>(2*L+1, 2*L+1, 2*L+1);

    for(arma::uword l = 0; l <= (arma::uword)L; l++)
    {
        arma::cx_mat It = I(span(l), span(L-l, L+l), span(L-l,L+l));
        
        arma::mat    W  = wig_d[l];
        arma::cx_cube TF_l = arma::zeros<arma::cx_cube>(2*L+1, 2*L+1,2*L+1);
        
        for(int h = -((int)l); h <= (int)l; h++)
            TF_l(span(L-l,L+l), span(h+L), span(L-l,L+l)) = (W.row(h+l).t() * W.col(h+l).t()) % It;
        
        TF += TF_l;
    }

    // Shrink for inverse FFT.
    TF = TF.subcube(0, 0, 0, TF.n_rows-2, TF.n_cols-2, TF.n_slices-2);

    arma::cx_cube cors = ifft(TF);
    
    arma::vec ang = arma::linspace(0, 2*M_PI, 2*L+1);
    ang = ang.subvec(0, ang.n_elem-2);

    arma::cube ang1, ang2, ang3;
    std::tie(ang1, ang2, ang3) = ndgrid(ang, ang, ang);
    //boost::tie(ang1, ang2, ang3) = ndgrid(ang, ang, ang);

    // -1i * L * (ang1 + ang2 + ang3)
    arma::cx_cube tmp(arma::zeros(2*L, 2*L, 2*L), - ((double)L) * (ang1 + ang2 + ang3));

    // n_elem * exp( - 1i * L * (ang1+ang2+ang3) ) % cors
    cors = cors.n_elem * arma::exp(tmp) % cors;

    return cors;
}


std::vector<arma::cx_mat> rot_search_expansion(const arma::cube &vol, unsigned int max_l, const std::vector<double> &radius, arma::vec3 center)
{
    std::vector<arma::cx_mat> coefs;

    // fill with zero if out of bounds.
    cubic_interpolater ci(vol, 0.0);

    for(size_t i = 0; i < radius.size(); i++)
    {
        /* We will represent the function values on the sphere, by sampling at
        * equally spaced points in angle space.

        Starting with theta in [0,pi] (latitude) and phi in [0,2*pi]
        (longitude), we can discritize into grid of size Nx2N.
        */

        /* The size of the matrix we will use.  Nx2N.  */
        unsigned int nlat = ceil(    M_PI * radius[i]);
        unsigned int nlon = ceil(2 * M_PI * radius[i]);

        // points on sphere are defined by two angles.
        arma::vec theta = arma::linspace<arma::vec>(0.0,     M_PI, nlat);
        arma::vec phi   = arma::linspace<arma::vec>(0.0, 2.0*M_PI, nlon);

        arma::mat surface = arma::zeros<arma::mat>(nlat, nlon);

        // for each spherical coordinate (r, theta, phi) find the cartesian (x,y,z) coordiate.
        //
        // interpolate the function sampled by vol at that point.
        arma::vec3 x;
        for(size_t j = 0; j < nlat; j++)
        {
            double st = sin(theta[j]), ct = cos(theta[j]);
            for(size_t k = 0; k < nlon; k++)
            {
                double sp = sin(phi[k]), cp = cos(phi[k]);
                x(0) = radius[i] * st * cp + center(0);
                x(1) = radius[i] * st * sp + center(1);
                x(2) = radius[i] * ct      + center(2);

                surface(j,k) = ci(x);
            }
        }

        unsigned int Lnyq = std::min( ceil( (nlon - 1.0)/2.0),nlat - 1.0);
        unsigned int L = std::min( Lnyq, max_l );

        // do spherical harmonic transform and return the coefficients.
        coefs.push_back(forward_sht(surface, L));
    }

    return coefs;
}



std::vector<std::tuple<double, int, int, int> > local_max_index(const arma::cube &score, unsigned int peak_spacing)
//std::vector<boost::tuple<double, int, int, int> > local_max_index(const arma::cube &score, unsigned int peak_spacing)
{

    std::vector<std::tuple<double, int, int, int> > score_coord;
    //std::vector<boost::tuple<double, int, int, int> > score_coord;

    // Dilate the cube to produce out.
    int se_width = score.n_rows / peak_spacing;

    if(se_width < 3) se_width = 3;
    if(se_width % 2 == 0) se_width ++;

    arma::cube out = dilate(score, se_width);
    
    for(size_t i = 0; i < out.n_rows; i++)
        for(size_t j = 0; j < out.n_cols; j++)
            for(size_t k = 0; k < out.n_slices; k++)
                if(out(i,j,k) == score(i,j,k))
                    score_coord.push_back(std::make_tuple(score(i,j,k), i, j, k));
                    //score_coord.push_back(boost::make_tuple(score(i,j,k), i, j, k));

    return score_coord;
}


std::tuple<std::vector<euler_angle>, std::vector<double> > local_max_angles(const arma::cube &score, unsigned int peak_spacing)
//boost::tuple<std::vector<euler_angle>, std::vector<double> > local_max_angles(const arma::cube &score, unsigned int peak_spacing)
{

    std::vector<std::tuple<double, int, int, int> > score_coord = local_max_index(score, peak_spacing);
    //std::vector<boost::tuple<double, int, int, int> > score_coord = local_max_index(score, peak_spacing);

    std::vector<euler_angle> coord;
    std::vector<double> vscore;

    for(size_t r = 0; r < score_coord.size(); r++)
    {
        double s = std::get<0>(score_coord[r]);
        //double s = boost::get<0>(score_coord[r]);
        int    i = std::get<1>(score_coord[r]);
        //int    i = boost::get<1>(score_coord[r]);
        int    j = std::get<2>(score_coord[r]);
        //int    j = boost::get<2>(score_coord[r]);
        int    k = std::get<3>(score_coord[r]);
        //int    k = boost::get<3>(score_coord[r]);

        // convert from xi, eta, omega to euler angles by rotation.
        // Kovacs & Wriggers (2002) pg. 1284 footnote.
        
        double ai = ((2*M_PI*i) / score.n_rows   - M_PI/2.0);
        double aj = ((2*M_PI*j) / score.n_cols   - M_PI);
        double ak = ((2*M_PI*k) / score.n_slices - M_PI/2.0);
        euler_angle ang(ai, aj, ak);

        coord.push_back(ang);
        vscore.push_back(s);
    }

    return std::make_tuple(coord, vscore);
    //return boost::make_tuple(coord, vscore);
}

bool tup_compare(const std::tuple<double, arma::vec3, euler_angle> &a, const std::tuple<double, arma::vec3, euler_angle> &b)
//bool tup_compare(const boost::tuple<double, arma::vec3, euler_angle> &a, const boost::tuple<double, arma::vec3, euler_angle> &b)
{
    // sort ascending. use >.
    return std::get<0>(a) > std::get<0>(b);
    //return boost::get<0>(a) > boost::get<0>(b);
}


std::vector<std::tuple<double, arma::vec3, euler_angle> > combined_search( const arma::cube &vol1, const arma::cube &mask1, const arma::cube &vol2, const arma::cube &mask2, unsigned int L)
//std::vector<boost::tuple<double, arma::vec3, euler_angle> > combined_search( const arma::cube &vol1, const arma::cube &mask1, const arma::cube &vol2, const arma::cube &mask2, unsigned int L)
{
    if( ! (arma::same_shape(vol1, vol2) && arma::same_shape(vol1, mask1) && arma::same_shape(vol1, mask2)) )
        throw fatal_error() << "combined_search: volumes and masks must all be same size.";

    // fft in 3d of volume.  
    arma::cx_cube fft1 = fft(vol1);
    arma::cx_cube fft2 = fft(vol2);

    // delete zero frequency coefficients so that the mean of real space values are zero
    // set 0,0,0 entry to zero before shift instead of mid_co after shift.
    fft1(0,0,0) = 0.0;
    fft2(0,0,0) = 0.0;

    // todo: clean up.
    //fft1 = fftshift(fft1);
    //fft2 = fftshift(fft2);
    //arma::cube fft1_abs = arma::abs(fft1);
    //arma::cube fft2_abs = arma::abs(fft2);

    arma::cube fft1_abs = arma::abs(fftshift(fft1));
    arma::cube fft2_abs = arma::abs(fftshift(fft2));


    // Create wigner D-matrices, used later.
    //! @note consider precomputation and loading from disk, or saving across combined_search runs.
    std::vector<arma::mat> wig_d = wigner_d(M_PI/2.0, L);


    // masks may be weights. need to be squared.
    // not necessarily 0/1.
    arma::cube mask1sq = mask1 % mask1;
    arma::cube mask2sq = mask2 % mask2;
    
    // one shell for every cube. N/2 shells.
    std::vector<double> radius(arma::max(arma::shape(mask1))/2.0);
    for(size_t i = 0; i < radius.size(); i++)
        radius[i] = (i+1.0);

    arma::vec3 mid_co = get_fftshift_center(vol1);

    arma::cx_cube cors_12 = rot_search_cor(fft1_abs % mask1sq, fft2_abs % mask2sq, L, radius, wig_d, mid_co);

    // denominator left part.
    arma::cx_cube sqt_cors_11 = arma::sqrt(rot_search_cor( fft1_abs % fft1_abs % mask1sq, mask2sq, L, radius, wig_d, mid_co));
    
    // denominator right part.
    arma::cx_cube sqt_cors_22 = arma::sqrt(rot_search_cor( mask1sq, fft2_abs % fft2_abs % mask2sq, L, radius, wig_d, mid_co));

    arma::cx_cube cors = cors_12 / (sqt_cors_11 % sqt_cors_22);

    // find local maximum in cors matrix. 
    // each element of cors corresponds to a different rotation angle.
    std::vector<euler_angle> angs;
    std::vector<double> dummy_scores;

    std::tie(angs, dummy_scores) = local_max_angles(arma::real(cors), 8);
    //boost::tie(angs, dummy_scores) = local_max_angles(arma::real(cors), 8);

    std::vector<std::tuple<double, arma::vec3, euler_angle> > angs_locs_scores;

    if(angs.size() == 0) // || angs.size() > 100)
    {
        //throw fatal_error() << "combined_search failed to find any angles, or too many...";
        return angs_locs_scores;
    }

    //std::cout << "angs.size() = " << angs.size() << std::endl;

    // remove redundant angles. 
    std::tie(angs, dummy_scores) = angle_list_redundancy_removal_zyz(angs, dummy_scores, 0.01);
    //boost::tie(angs, dummy_scores) = angle_list_redundancy_removal_zyz(angs, dummy_scores, 0.01);

    // locs_r will be a list of displacements that are optimal for each given
    // angle from ang.  Translation is (x,y,z) displacement. 
    std::vector<arma::vec3> locs_r(angs.size());

    std::vector<double> scores(angs.size());

    // We will return the list of best matches in a tuple of : (score, trans, rot)
    // where the given translation/rotation will give the correlation score.
    for(size_t i = 0; i < angs.size(); i++)
    {
        std::tie(locs_r[i], scores[i]) = cons_corr_max(vol1, mask1, vol2, mask2, angs[i]);
        //boost::tie(locs_r[i], scores[i]) = cons_corr_max(vol1, mask1, vol2, mask2, angs[i]);
        angs_locs_scores.push_back(std::make_tuple(scores[i], locs_r[i], angs[i]));
        //angs_locs_scores.push_back(boost::make_tuple(scores[i], locs_r[i], angs[i]));
    }
    
    // sort the list by decreasing score.
    std::sort(angs_locs_scores.begin(), angs_locs_scores.end(), tup_compare);
    
    // Can this ever happen?  Or will it always be caught above?
    if(angs_locs_scores.size() == 0)
    {
        throw fatal_error() << "combined_search failed to find any matches";
    }    
    return angs_locs_scores;
}

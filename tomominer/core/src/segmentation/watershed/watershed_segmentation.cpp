


#include <iostream>
#include <vector>
#include <set>
#include <map>
#include <queue>
#include <cassert>

#include "arma_extend.hpp"
#include "watershed_segmentation.hpp"

class IntensityLocation {
    public: 
        double _intensity;
        int _loc;

        IntensityLocation(double intensity, int loc) {
            _intensity = intensity;
            _loc = loc;
        }
};


class CompatreIntensityLocation {

    public:
        // Returns true if i1 has a smaller intensity than i2
        bool operator() (const IntensityLocation& i1, const IntensityLocation& i2) {
            return (i1._intensity < i2._intensity);
        }
};


std::vector<int> neighbor_inds(const arma::uvec &shape, int i) {
    
    arma::uvec sub = arma::ind2sub(shape, i);

    std::vector<int> n;
    for(int i0=-1; i0<=1; i0++)
        for(int i1=-1; i1<=1; i1++)
            for(int i2=-1; i2<=1; i2++) {
                if((i0==0) && (i1==0) && (i2==0))   continue;

                int j0 = sub(0) + i0;       if( (j0<0) || (j0>=int(shape(0))) )  continue;
                int j1 = sub(1) + i1;       if( (j1<0) || (j1>=int(shape(1))) )  continue;
                int j2 = sub(2) + i2;       if( (j2<0) || (j2>=int(shape(2))) )  continue;

                arma::uvec sub_j(3);
                sub_j(0) = j0;      sub_j(1) = j1;      sub_j(2) = j2;

                n.push_back(arma::sub2ind(shape, sub_j));
            }

    return n;            
}

int watershed_segmentation(const arma::cube &vol, const arma::icube &lbl, const unsigned int max_overall_voxel_num, const unsigned int max_segment_voxel_num, const int queue_label, const int conflict_lbl, arma::icube &vol_seg_lbl, arma::ucube &vol_num, arma::ucube &overall_num)
{
    //std::cout << "watershed_segmentation()" << std::endl;

    assert(queue_label < 0);
    assert(conflict_lbl < 0);
    int unlabel_label = 0;


    arma::uvec3 siz = arma::shape(vol);

/*
    // a simple test of correctness of ind2sub and sub2ind
    for(unsigned int i=0; i<vol.n_elem; i++) {
        arma::uvec sub = arma::ind2sub(siz, i);
        assert( vol(sub(0), sub(1), sub(2)) == vol(i));

        int it = arma::sub2ind(siz, sub);
        assert( vol(it) == vol(i) );
    }
*/

    // make a copy of labels
    for(unsigned int i=0; i<lbl.n_elem; i++)     vol_seg_lbl(i) = lbl(i);
    
    unsigned int overall_num_count = 0;
    std::map <unsigned int, unsigned int> lbl_voxel_num;

    std::priority_queue<IntensityLocation, std::vector<IntensityLocation>, CompatreIntensityLocation> pq;     

    //finding seeds and add their neighbors into queue
    for(unsigned int i=0; i<lbl.n_elem; i++) {
        if(vol_seg_lbl(i) <= 0)     continue;

        std::vector<int> nei = neighbor_inds(siz, i);

        for(unsigned int nei_i=0; nei_i<nei.size(); nei_i++) {
            int j = nei[nei_i];

            if(!arma::is_finite(vol(j)))    continue;
            if(vol_seg_lbl(j) != unlabel_label)    continue;    

            vol_seg_lbl(j) = queue_label;

            pq.push(IntensityLocation(vol(j), j));
        }

        if(lbl_voxel_num.find(vol_seg_lbl(i)) == lbl_voxel_num.end()) {
            lbl_voxel_num[vol_seg_lbl(i)] = 1;
        } else {
            lbl_voxel_num[vol_seg_lbl(i)] ++;
        }
        vol_num(i) = lbl_voxel_num[vol_seg_lbl(i)];
        
        overall_num_count ++;
        overall_num(i) = overall_num_count;
        if(overall_num_count > max_overall_voxel_num)       break;
    }

    
    while(!pq.empty()) {
        IntensityLocation il = pq.top();        pq.pop();
        int i = il._loc;

        //std::cout << vol(i) << "\t";

        assert(vol_seg_lbl(i) == queue_label);
        assert(il._intensity == vol(i));

        lbl_voxel_num[vol_seg_lbl(i)] ++;
        vol_num(i) = lbl_voxel_num[vol_seg_lbl(i)];
        if(vol_num(i) > max_segment_voxel_num)  continue;

        overall_num_count ++;
        overall_num(i) = overall_num_count;

        if(overall_num_count > max_overall_voxel_num)       break;

        std::vector<int> nei = neighbor_inds(siz, i);
        std::set<int> nei_lbl;
        for(unsigned int nei_i=0; nei_i<nei.size(); nei_i++) {
            int j = nei[nei_i];
            if(vol_seg_lbl(j) <= 0) continue; 

            nei_lbl.insert(vol_seg_lbl(j));
        }

        assert (nei_lbl.size() > 0);

        if(nei_lbl.size() == 1) {
            // extract the element from nei_lbl
            std::set<int>::iterator it = nei_lbl.begin();
            vol_seg_lbl(i) = *it;

            // add unlabeled neighbors to the queue
            for(unsigned int nei_i=0; nei_i<nei.size(); nei_i++) {
                int j = nei[nei_i];

                if(!arma::is_finite(vol(j)))    continue;
                if(vol_seg_lbl(j) != unlabel_label)    continue;

                vol_seg_lbl(j) = queue_label;
                pq.push(IntensityLocation(vol(j), j));
            }

        } else {
            vol_seg_lbl(i) = conflict_lbl;

            // if i is at the intersection of two segments, do not process its neighbors
        }

    }


    return 1;       // this indicates successful
}




 
int segment_boundary(const arma::icube &lbl, arma::icube &bdry)
{

    for(unsigned int i=0; i<bdry.n_elem; i++)     bdry(i) = 0;

    int count = 0;
    arma::uvec3 siz = arma::shape(lbl);
    for(unsigned int i=0; i<bdry.n_elem; i++) {
        std::vector<int> nei = neighbor_inds(siz, i);

        for(unsigned int nei_i=0; nei_i<nei.size(); nei_i++) {
            int j = nei[nei_i];
            
            if(lbl(i) != lbl(j)) {
                bdry(i) = 1;
                count ++;
                break;
            }
        }
    }

    return count;   // this indicates successful

}


// given a binary mask that only contain zeros and positive numbers, find and label all connected regions
int connected_regions(const arma::icube &msk, arma::icube &lbl)
{
    for(unsigned int i=0; i<lbl.n_elem; i++)     lbl(i) = 0;

    arma::uvec3 siz = arma::shape(msk);

    std::queue<unsigned int> q;

    int current_lbl = 0;
    while(true) {
        bool found_new_label = false;
        current_lbl ++;

        for(unsigned int i=0; i<msk.n_elem; i++) {
            if( (msk(i) > 0) && (lbl(i) == 0) ) {
                lbl(i) = current_lbl;
                q.push(i);

                found_new_label = true;
                break;
            }
        }

        if(!found_new_label)    break;
        
        while(!q.empty()) {
            int i = q.front();            q.pop();

            assert(lbl(i) == current_lbl);

            std::vector<int> nei = neighbor_inds(siz, i);

            for(unsigned int nei_i=0; nei_i<nei.size(); nei_i++) {
                unsigned int j = nei[nei_i];
                if(lbl(j) > 0) {
                    continue;
                }

                if(msk(j) == msk(i)) {
                    lbl(j) = lbl(i);
                    q.push(j);
                }
            }

            
        }

    }

    return (current_lbl-1);
}


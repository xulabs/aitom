// code adapted from 
// http://www.mathworks.com/matlabcentral/fileexchange/24998-2d3d-image-segmentation-toolbox

#ifndef __ZY_ac_distance_transform_3d_hpp__
#define __ZY_ac_distance_transform_3d_hpp__


#include "BinaryHeap.h"
#include <memory.h>
#include <math.h>

enum state {DEAD, ACTIVE, INACTIVE, BOUNDARY, SEED}; 

template<class InputType, class OutputType>
void DistanceTransform3D(const InputType* pIn, OutputType* D, const unsigned int* dims)
{
	int N = dims[0]*dims[1]*dims[2]; 
	CBinaryHeapMinSort<int, OutputType> heap(N);
	unsigned char* state = new unsigned char[N]; 

	int plane_size = dims[0]*dims[1]; 
	int row_size = dims[0]; 
	int neighbor_idx[6] = {-1, 1, -row_size, row_size, -plane_size, plane_size}; 

	// (1) Put seed into the heap and set their states to active, others set to inactive
	// NOTE: for efficiency purpose, boudaries of the matrix will not be processed. 
	memset(state, BOUNDARY, N); 
	for(int k = 1; k < (int)dims[2]-1; k++) {				
		for(int j = 1;  j < (int)dims[1]-1; j++) {
			int idx_base = k*plane_size + j*row_size; 
			for(int i = 1, idx = idx_base + 1; i < (int)dims[0]-1;  i++, idx++) {				
				if( pIn[idx] == 1 ) {
					heap.Insert(idx, 0); 
					D[idx] = 0; 
					state[idx] = SEED; 
				}
				else
					state[idx] = INACTIVE;
			}
		}
	}

	// (2) Extract a point from the heap
	int idx; 
	OutputType weight; 
	while( heap.Extract(&idx, &weight) )
	{
		// (3) if the state of the point is active ...
		if( state[idx] == DEAD ) continue; 		
		// (4) calculat the distance form its dead neighbors by solving ||\nabla u|| = 1; 
#define INF 10000000
		OutputType a, b, c, v1, v2, delta; 
		int idx1, idx2; 
		if( state[idx] != SEED )
		{
			a = b = 0;  c = -1; 	

			idx1 = idx + neighbor_idx[0]; 
			idx2 = idx + neighbor_idx[1]; 
			v1 = (state[idx1] == DEAD) ? D[idx1] : INF; 
			v2 = (state[idx2] == DEAD) ? D[idx2] : INF; 
			if( v1 > v2 ) v1 = v2; 
			if( v1 != INF ) { a += 1; b += -2*v1; c += v1*v1; }

			idx1 = idx + neighbor_idx[2]; 
			idx2 = idx + neighbor_idx[3]; 
			v1 = (state[idx1] == DEAD) ? D[idx1] : INF; 
			v2 = (state[idx2] == DEAD) ? D[idx2] : INF; 
			if( v1 > v2 ) v1 = v2; 
			if( v1 != INF ) { a += 1; b += -2*v1; c += v1*v1; }

			idx1 = idx + neighbor_idx[4]; 
			idx2 = idx + neighbor_idx[5]; 
			v1 = (state[idx1] == DEAD) ? D[idx1] : INF; 
			v2 = (state[idx2] == DEAD) ? D[idx2] : INF; 
			if( v1 > v2 ) v1 = v2; 
			if( v1 != INF ) { a += 1; b += -2*v1; c += v1*v1; }

			// NOTE: a shouldn't be 0!
			delta = b*b - 4*a*c;
			//mexPrintf("delta = %f\n", delta);
			D[idx] = ( delta>0 ) ? .5*(- b + sqrt(delta))/a : 0; 
		}
		else
			D[idx] = 0; 
		// (5) set the state to dead
		state[idx] = DEAD; 

		// (6) search the 6-neighborhoods for non-deads (a) calcualte the distance (b) push them to the heap (c) set status to active
		for(int i = 0; i < 6; i++)
		{			
			int n_idx = idx + neighbor_idx[i]; 
			if( state[n_idx] == INACTIVE || state[n_idx] == ACTIVE )
			{
				heap.Insert(n_idx, D[idx] + 1);  
				state[n_idx] = ACTIVE; 
			}
		}
	}

	delete [] state; 
}


#endif // __ZY_ac_distance_transform_3d_hpp__

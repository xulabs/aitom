// code from
// http://www.mathworks.com/matlabcentral/fileexchange/24998-2d3d-image-segmentation-toolbox

#ifndef __ZY_CBinaryHeap_H__
#define __ZY_CBinaryHeap_H__

#include <cassert>
#include <exception>
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	 CBinaryHeap
//   BinaryHeap implementation 
//
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class IndexType, class WeightType> 
class CBinaryHeapSort
{
public:
	virtual bool Insert(IndexType index, WeightType weight) = 0;
	virtual bool Extract(IndexType *index, WeightType *weight) = 0;
	virtual bool IsEmpty() = 0;
	virtual void ReplaceData(IndexType idx, WeightType weight) = 0;
	virtual ~CBinaryHeapSort() {}; // This line is extremly important if you want destruction functions (releasing memories!) of the derived classes to work properly. 
}; 


template<class IndexType, class WeightType> 
class CBinaryHeapMinSort: public CBinaryHeapSort<IndexType, WeightType>
{
private: 
	struct SPair
	{
		IndexType idx; 
		WeightType weight; 
	}; 

private:
	SPair* m_heap=NULL; 
	int m_heap_idx; // indicates the heap position for next insertion.
	int m_heap_size; 
public:
	CBinaryHeapMinSort(int number_of_points)
	{
		m_heap = new SPair[number_of_points]; 			
		m_heap_idx = 0; 
		m_heap_size = number_of_points; 
	}
	~CBinaryHeapMinSort()
	{
		if(m_heap) delete []m_heap; 
	}
public:
	bool Insert(IndexType index, WeightType weight)
	{
                if( m_heap_idx >= m_heap_size ) {
                    return false;
                }

		m_heap[m_heap_idx].idx = index; 
		m_heap[m_heap_idx].weight = weight; 

		int idx = m_heap_idx; 
		m_heap_idx++; 
		if( m_heap_idx >= m_heap_size ) {
                    
                    return false;

                    std::cerr << "Insert() m_heap_idx >= m_heap_size " << m_heap_idx << std::endl;
                    throw std::exception(); 
                }

		while(idx != 0)
		{
			int parent_idx = int((idx-1)/2); 
			if( m_heap[idx].weight < m_heap[parent_idx].weight )
			{
				SPair tmp = m_heap[idx]; 
				m_heap[idx] = m_heap[parent_idx]; 
				m_heap[parent_idx] = tmp; 

				idx = parent_idx; 
			}
			else break; 
		}

                return true;
	}

	bool Extract(IndexType *index, WeightType *weight)
	{
		if( this->IsEmpty() ) return false; 

		*index = m_heap[0].idx; 
		*weight = m_heap[0].weight; 

		m_heap_idx --; 
		int idx = m_heap_idx; 
		if( idx > 0) // if still exists node in the heap
		{
			m_heap[0] = m_heap[idx]; // replace the first node by the last node
			idx = 0; 
			while(1)
			{
				int child_idx0 = 2*idx + 1; 
				int child_idx1 = child_idx0 +1; 
				if( child_idx0 >= m_heap_idx )  break; // no children for idx, i.e. idx has reached a leave, break. 
				else
				{
					int child_idx = child_idx0; 
					if( child_idx1 < m_heap_idx ) // if idx has two children.
						if( m_heap[child_idx].weight > m_heap[child_idx1].weight ) // find the child with larger value. 
							child_idx = child_idx1; 
					
					// if idx has smaller value than the child found, swap.
					if( m_heap[idx].weight > m_heap[child_idx].weight ) 
					{
						SPair tmp = m_heap[idx]; 
						m_heap[idx] = m_heap[child_idx]; 
						m_heap[child_idx] = tmp; 
						idx = child_idx; 
					}
					else break; // otherwise idx is in the right place, break; 
				}
			} 
		}

		return true; 
	}

	bool IsEmpty()
	{
		return m_heap_idx == 0; 
	}

	void ReplaceData(IndexType idx, WeightType weight)
	{
		Insert(idx, weight); 
	}

	void Reset()
	{
		m_heap_idx = 0; 
	}
}; 

#endif // __ZY_CBinaryHeap_H__

// code adapted from 
// http://www.mathworks.com/matlabcentral/fileexchange/24998-2d3d-image-segmentation-toolbox


#ifndef __ZY_binary_boundary_detection_hpp__
#define __ZY_binary_boundary_detection_hpp__

#include <assert.h>
#include <iostream>


void zy_BinaryBoundaryDetection(char *pI, int width, int height, int depth, int type, char *pOut)
{
	int inv_type = !type;
	int i, j, k, idx, idx0, count;
	int on_boundary = 0;
	int wh = width*height;





	count = 0;
	idx = 1;
	for(i=1; i<depth-1; i++)
	{
		idx0 = wh*i;
		for(j=1; j<width-1; j++)
		{
			idx = idx0 + j*height;
			for(k=1; k<height-1; k++)
			{
				idx++;
				/* If the pixel is 'on', check if its neighbors are 'on'
				  if negative, the pixel is considered to be on boundary. */
				if(pI[idx] == type)
				{
					if(pI[idx-1] == inv_type) pOut[idx] = 1;
					else
					{
						if(pI[idx+1] == inv_type) pOut[idx] = 1;
						else
						{
							if(pI[idx-height] == inv_type) pOut[idx] = 1;
							else
							{
								if(pI[idx+height] == inv_type) pOut[idx] = 1;
								else
								{
									if(pI[idx-wh] == inv_type) pOut[idx] = 1;
									else
									{
										if(pI[idx+wh] == inv_type) pOut[idx] = 1;
									}
								}
							}
						}
                    }
				}
			}
		}
	}
}

#endif // __ZY_binary_boundary_detection_hpp__


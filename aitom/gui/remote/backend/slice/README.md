# Slicer
## Introduction
Mrc slicer
## Usage
In `slice.py`, specify a mrc file, rotation center and rotation angle(in degree)    
```python
if __name__ == '__main__':
    mrc = mrcfile.open('test.mrc')
    model = mrc.data
    # 73 32 -47
    #x_rot, y_rot, z_rot = 12, -23, 83
    x_rot, y_rot, z_rot = 73, 32, -47
    center = (100, 100, 100)
    slice3d(model, center, x_rot, y_rot, z_rot)
```
A slice surface image(in cube) and a slice image will be shown.   
## Implementation details
- Searching Method: neighbor serach(binary search is less effective)
- Interpolation Method: nearest neighbor interpolation

## Future works
- Add web support

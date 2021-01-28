# Filament Tracing
An automated way of segmenting actin filaments published by Rigort et al. (2012) is implemented here. This method is based on template matching and a recursive tracing algorithm. From template matching, a cross-correlation matrix and three template rotation matrices (using ZYZ rotations) are generated. The matrices are used as inputs to the tracing algorithm to produce a binary mask storing the tracing result.


## Template Matching
For template matching, see our [tutorial](https://github.com/xulabs/aitom/blob/master/doc/tutorials/007_template_matching_tutorial.md) and [code](https://github.com/xulabs/aitom/blob/master/doc/tutorials/007_template_matching.py), which would output four .npy files storing the correlation matrix and the template rotation matrices.

## Tracing Algorithm
The tracing algorithm is based on a similarity function that takes into account the cross-correlation value at a voxel from template matching, the smoothness, the linearity, and the distance coefficients at the next voxel. The user can adjust the tracing result by using different threshold and parameter values. For a detailed description of the tracing algorithm, see Rigort et al. (2012).

### Input Parameters
The following parameters are needed for the tracing algorithm:
- t1: threshold for similarity, defualt=0.000001
- t2: threshold for correlation coefficient, default= 0.0007
- sigma_c: measure for smoothness, default=1.0 (smaller sigma_c favors smoother lines)
- sigma_l: measure for linearity, default=1.0 (smaller sigma_l favors straight lines)
- sigma_d: measure for distance, default=1.0 (smaller sigma_d restricts line extension)
- sc_path: file path of [the search cone](https://cmu.box.com/s/eg0tr9m1jkar1wsgjmkajgqqbcytrlt2)
- out_path: output file path (.mrc/.rec) for the tracing result (a binary mask)
- c_path: file path of the correlation matrix (e.g. c.npy from [007_template_matching.py](https://github.com/xulabs/aitom/blob/master/doc/tutorials/007_template_matching.py))
- phi_path, theta_path, psi_path: file paths of the ZYZ rotation matrices (e.g. phi.npy, theta.npy, and psi.npy from 007_template_matching.py)

To run the tracing algorithm:
```python
import aitom.segmentation.curve.filament_tracing.tracing as FT
t1 = 0.000001 #threshold for similarity
t2 = 0.0007 #threshold for correlation coefficient
sigma_c = 1.0 #smootheness
sigma_l = 1.0 #linearity
sigma_d = 1.0 #distance
sc_path = './search_cone.rec' #TODO: change filepaths
out_path = './bm.rec'
c_path = './results/test_id-c.npy'
phi_path = './results/test_id-phi.npy'
theta_path = './results/test_id-theta.npy'
psi_path = './results/test_id-psi.npy'
result = FT.trace(t1, t2, sigma_c, sigma_l, sigma_d, sc_path, out_path, c_path, phi_path, theta_path, psi_path)
```

### Search Cone
The search cone used for actin filaments can be found at: https://cmu.box.com/s/eg0tr9m1jkar1wsgjmkajgqqbcytrlt2. The search cone given is a binary mask of a fixed cylinder (height = 5, width = 10). This tracing method can also be applied to other linear structures such as microtubes with custom search cones of different sizes.

## References
Rigort, A., Günther, D., Hegerl, R., Baum, D., Weber, B., Prohaska, S., Medalia, O., Baumeister, W., & Hege, H. C. (2012). Automated segmentation of electron tomograms for a quantitative description of actin filament networks. Journal of structural biology, 177(1), 135–144. https://doi.org/10.1016/j.jsb.2011.08.012

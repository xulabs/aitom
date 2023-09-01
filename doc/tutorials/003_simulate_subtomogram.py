'''
This script shows an example of how to
1. generate a density map containing a toy structure
2. randomly rotate and translate the map
3. convert the map to subtomogram

IMPORTANT: If you are going to use this simulated data for deep learning, please do not save the simulated data to disk.
Because simated data will consume large amount of storage. Please construct a generator to directly feed the
simulated data to your model from memory.
'''

import aitom.io.file as IF
import numpy as N
# Please Note that `aitom_core` module is in the server, not public for now
import aitom_core.simulation.reconstruction.reconstruction__simple_convolution as TSRSC
import aitom.model.util as MU
import aitom.geometry.ang_loc as GAL
import aitom.geometry.rotate as GR
import argparse
import scipy.stats as SS
import aitom.image.io as IIO
import aitom.image.vol.util as IVU

class SimulatedSubtomogramGenerator():
  def __init__(self,output_dir,op,loc_proportion):
    self.output_dir = output_dir
    self.op = op
    self.loc_proportion = loc_proportion 
  
  # generate a density map v that contains a toy structure
  def Generate_density_map(self):
    v = MU.generate_toy_model(dim_siz=64)  # generate a pseudo density map
    print(f"\nsize of v = {v.shape}\n")
    return v

  # randomly rotate and translate v
  def rotate_translate_map(self, v):
    loc_max = N.array(v.shape, dtype=float) * self.loc_proportion
    angle = GAL.random_rotation_angle_zyz()
    loc_r = (N.random.random(3)-0.5)*loc_max
    vr = GR.rotate(v, angle=angle, loc_r=loc_r, default_val=0.0)
    return vr

  # generate simulated subtomogram vb from v
  def generate_subtomogram(self,vr):
    vb = TSRSC.do_reconstruction(vr, self.op, verbose=True)
    print('\nvb', 'mean', vb.mean(), 'std', vb.std(), 'var', vb.var(),"\n")
    return vb
  
  # save v and vb as 3D grey scale images
  def save_result(self,v,vb,vr):
    IF.put_mrc(vb, f'{self.output_dir}/vb.mrc', overwrite=True)
    IF.put_mrc(v, f'{self.output_dir}/v.mrc', overwrite=True)
    IIO.save_png(IVU.cub_img(vb)['im'], f'{self.output_dir}/vb.png')
    IIO.save_png(IVU.cub_img(v)['im'], f'{self.output_dir}/v.png')

    # verify the correctness of SNR estimation
    vb_rep = TSRSC.do_reconstruction(vr, self.op, verbose=True)
    # calculate SNR
    vb_corr = SS.pearsonr(vb.flatten(), vb_rep.flatten())[0]
    vb_snr = 2*vb_corr / (1 - vb_corr)
    print('\nSNR', 'parameter', self.op['model']['SNR'], 'estimated', vb_snr)  # fsc = ssnr / (2.0 + ssnr)


def main():
    parser = argparse.ArgumentParser(description="Generate simulated subtomograms")
    parser.add_argument("output_dir", type=str, help="Output directory for saving generated data")
    parser.add_argument("--missing_wedge_angle", type=float, default=30, help="Missing wedge angle")
    parser.add_argument("--snr", type=float, default=0.05, help="Signal-to-Noise Ratio (SNR)")
    parser.add_argument("--pix_size", type=float, default=1.0, help="Pixel size")
    parser.add_argument("--Dz", type=float, default=-5.0, help="Defocus value")
    parser.add_argument("--voltage", type=int, default=300, help="Accelerating voltage")
    parser.add_argument("--Cs", type=float, default=2.0, help="Spherical aberration coefficient")
    parser.add_argument("--sigma", type=float, default=0.4, help="Standard deviation")
    parser.add_argument("--loc_proportion", type=float, default=0.1, help="Maximum Translation")
    args = parser.parse_args()

    op = {
        'model': {'missing_wedge_angle': args.missing_wedge_angle, 'SNR': args.snr},
        'ctf': {'pix_size': args.pix_size, 'Dz': args.Dz, 'voltage': args.voltage, 'Cs': args.Cs, 'sigma': args.sigma}
    }

    generator = SimulatedSubtomogramGenerator(args.output_dir, op, args.loc_proportion)
    v = generator.Generate_density_map()
    vr = generator.rotate_translate_map(v)
    vb = generator.generate_subtomogram(vr)
    generator.save_result(v, vb,vr)

if __name__ == "__main__":
  main()

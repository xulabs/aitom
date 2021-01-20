import sys
import tomominer.simulation.reconstruction__simple_convolution as TSRSC  # if use 171 server
# sys.path.append('/shared/src/aitom/aitom_core')      #if use gpu0 server
# import aitom_core.simulaiton.reconstruction as TSRSC #if use gpu0 server

sys.path.append("..")

op = {'model': {'missing_wedge_angle': 30, 'SNR': 0.4},
      'ctf': {'pix_size': 1.0, 'Dz': -5.0, 'voltage': 300, 'Cs': 2.0, 'sigma': 0.4}}


def map2tomo(map, op):
    vb = TSRSC.do_reconstruction(map, op, verbose=True)
    print('vb', 'mean', vb.mean(), 'std', vb.std(), 'var', vb.var())
    return vb


if __name__ == '__main__':
    import iomap as IM

    packmap = IM.readMrcMap('../IOfile/packmap/mrc/packmap1.mrc')
    vb = map2tomo(packmap, op)
    IM.map2mrc(vb, '../IOfile/tomo/mrc/tomo_SNR04.mrc')
    IM.map2png(vb, '../IOfile/tomo/png/tomo_SNR04.png')

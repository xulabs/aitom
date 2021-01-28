"""
Computing is synchronized.
It should be changed to an asynchronized version later.
"""

from .proto import AESingleRequest, AESingleResponse, AEResultRequest, AEResultResponse
from .particle_picking_and_autoencoder_util import mkdir, view_clusters
from .pool import particlePickingPool
import aitom.classify.deep.unsupervised.autoencoder.autoencoder as AE
import aitom.io.file as AIF
import os
import base64


def autoencoder_single_main(request: AESingleRequest):
    item = particlePickingPool.get(request.path)

    remove_particles = list(map(int, request.remove_particles.replace(',', ' ').split()))
    pick_num = 100
    item.pick.select(remove_particles, pick_num)
    subvols_loc = os.path.join(
        item.dump_folder, "selected_demo_single_particle_subvolumes.pickle")
    output_dir = os.path.join(item.dump_folder, 'autoencoder_particle')

    d = AIF.pickle_load(subvols_loc)
    AE.encoder_simple_conv_test(d=d, pose=None, img_org_file=False, out_dir=output_dir, clus_num=1)
    AE.kmeans_centers_plot(AE.op_join(output_dir, 'clus-center'))
    return AESingleResponse()


def autoencoder_result_main(request: AEResultRequest):
    item = particlePickingPool.get(request.path)
    view_clusters(os.path.join(item.dump_folder, 'autoencoder_particle'))
    with open('tmp_ae.png', 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    res = AEResultResponse()
    res.img = b64
    return res

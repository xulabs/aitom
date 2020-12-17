from aitom.bin.picking import picking

from .proto import PPRequest, PPResponse, PPVisRequest, PPVisResponse, \
    PPResumeRequest, PPResumeResponse
from .particle_picking_and_autoencoder_util import mkdir, ParticlePicking
from .pool import particlePickingPool
import base64


def particle_picking_main(p: PPRequest):
    item = particlePickingPool.get(p.path, new_one=True)
    # print('item is', item.__dict__)
    result = picking(item.mrc_path, s1=p.sigma1, s2=p.sigma1 * 1.1, t=3, find_maxima=False,
                     partition_op=None, multiprocessing_process_num=0)
    ppr = PPResponse()
    ppr.pick_total = len(result)
    ppr.uid = item.uid

    item.pick.dump_subvol(result)
    item.add_proto(p)
    return ppr


def particle_picking_visualization_main(p: PPVisRequest):
    item = particlePickingPool.get(p.path)
    if p.subvol_num == -1:
        # view tomograph
        result = item.pick.view_subtom(p.subvol_num)
    else:
        # view subtomograph
        result = item.pick.view_subtom(p.subvol_num)
    with open(result, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    ppvr = PPVisResponse()
    ppvr.subvol_url = b64
    return ppvr


def particle_picking_resume_main(p: PPResumeRequest):
    pprr = PPResumeResponse()
    pprr.resumable_list = particlePickingPool.make_list()
    return pprr

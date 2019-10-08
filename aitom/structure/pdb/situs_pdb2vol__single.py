#!/usr/bin/env python


'''
convert a single PDB file to density map


~/ln/tomominer/tomominer/structure/pdb/situs_pdb2vol__single.py
'''

def convert(op):

    import aitom.structure.pdb.situs_pdb2vol as SP

    r = SP.convert(op)

    v = r['map']

    if 'out_map_size' in op:
        import tomominer.image.vol.util as IVU
        v = IVU.resize_center(v=v, s=op['out_map_size'], cval=0.0)

    import tomominer.io.file as IF
    IF.put_mrc(v, op['out_file'])



if __name__ == '__main__':

    import json
    with open('situs_pdb2vol__single__op.json') as f:       op = json.load(f)

    convert(op)


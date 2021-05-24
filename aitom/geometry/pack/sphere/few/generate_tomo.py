# change the parameter in op to the value you want.
# Change the input value in line 19 to the number of targets and modify the PDB id in line 21-30.
# change the folder of input densitymap in line 35
# Change the number in line 43 to the number of subtomograms for each target.
# change the save path in line 56-59 to the folder you want.

op = {
    'map':{'situs_pdb2vol_program':'/shared/opt/local/img/em/et/util/situs/Situs_2.7.2/bin/pdb2vol',
           'spacing_s': [10.0], 'resolution_s':[10.0],
           'pdb_dir':'IOfile/pdbfile/',
           'out_file':'/IOfile/map_single/situs_maps.pickle',
           'map_single_path': './IOfile/map_single'},
    'tomo':{'model':{'missing_wedge_angle':30, 'SNR':0.3},
            'ctf':{'pix_size':1.0, 'Dz':-5.0, 'voltage':300, 'Cs':2.0, 'sigma':0.4}},
    'target_size':32,
    'v': None
    }

for pdb_index in range(1):
    packing_op = {}
    if pdb_index == 0:
        packing_op['target'] = '1bxn'
    elif pdb_index == 1:
        packing_op['target'] = '1f1b'
    elif pdb_index == 2:
        packing_op['target'] = '1yg6'
    elif pdb_index == 3:
        packing_op['target'] = '2byu'
    else:
        packing_op['target'] = '4d4r'

    pdbid = packing_op['target']

    # read density map from mrc
    rootdir = 'IOfile/test/' + pdbid + '/densitymap_mrc'
    import map_tomo.iomap as IM
    print('density map file path:',rootdir)
    v = IM.readMrcMapDir(rootdir)
    print('read all density map done')

    # convert to tomo
    num = 0
    for num in range(1):
        print('index =',pdb_index, 'num =', num, pdbid, 'SNR = ', op['tomo']['model']['SNR'], 'start')

        # get one densitymap file
        filename = 'packtarget' + str(num)
        print('filename:',filename)
        target_packmap = v[filename]

        output = {
            'packmap': {
                'target': {
                    'mrc': 'IOfile/test/' + pdbid + '/map{}.mrc'.format(num),
                    'png': 'IOfile/test/' + pdbid + '/map{}.png'.format(num)}},
            'tomo': {
                'target': {
                    'mrc': 'IOfile/test/' + pdbid + '/tomo{}.mrc'.format(num),
                    'png': 'IOfile/test/' + pdbid + '/tomo{}.png'.format(num)}},
            'json': {
                'pack': 'IOfile/test/' + pdbid + '/packing{}.json'.format(num),
                'target': 'IOfile/test/' + pdbid + '/target{}.json'.format(num)}}

        import map_tomo.map2tomogram as MT
        target_tomo = MT.map2tomo(target_packmap, op['tomo'])
        print('convert to tomo')
        import map_tomo.iomap as IM
        IM.map2mrc(target_tomo, output['tomo']['target']['mrc'])
        print('save tomo')

        print('Done\n')
        num = num + 1

print('all Done')



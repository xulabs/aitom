"""
Code automatically generated with cleaning of non-relevant contents
Please cite: Xu et al. De novo visual proteomics of single cells through pattern mining
"""

import os, sys, shutil, time, copy, random, json, uuid
from collections import defaultdict
import pickle as pickle
import numpy as N
import aitom.tomominer.pursuit.multi.util as CU
import aitom.tomominer.io.file as IV


def fsc_stat_json_convert(fsc_stat):
    fsc_stat_t = {}
    for (pass_i, fsc_stat_) in fsc_stat.items():
        fsc_stat_t[int(pass_i)] = {}
        for (c, fsc_stat__) in fsc_stat_.items():
            fsc_stat_t[int(pass_i)][int(c)] = fsc_stat__
    return fsc_stat_t


def stop_test_stat(op, fsc_stat, pass_i, cluster_modes):
    if op['options']['stopping_test']['criterion'] == 0:
        selected_templates_max_pass_i = 0
        for (_, _fsc_stat) in fsc_stat.items():
            for (__, __fsc_stat) in _fsc_stat.items():
                if __fsc_stat['is_specific'] is not None:
                    continue
                if __fsc_stat['pass_i'] > selected_templates_max_pass_i:
                    selected_templates_max_pass_i = __fsc_stat['pass_i']
        selected_templates_min_pass_i = N.min(
            [_fsc_stat['pass_i'] for (_, _fsc_stat) in fsc_stat[pass_i].items() if (_fsc_stat['is_specific'] is None)])
        del _, _fsc_stat, __, __fsc_stat
        print('max pass of selected non-redundant average', selected_templates_max_pass_i)
    else:
        raise Exception('options_stopping_test_criterion')
    should_stop = False
    if (pass_i - selected_templates_max_pass_i) >= op['cluster']['stopping_test_pass_num']:
        should_stop = True
    cluster_modes = copy.deepcopy(cluster_modes)
    if should_stop:
        if ('adaptive_k_ratio' in op['cluster']['kmeans']) and ('kmeans-adaptave' not in set(cluster_modes)):
            print('include kmeans-adaptave mode for cluster population at next iteration')
            cluster_modes.append('kmeans-adaptave')
            should_stop = False
        if ('sequential_expansion' in op['cluster']) and ('sequential' not in set(cluster_modes)):
            print('include sequential clustering mode for cluster population at next iteration')
            cluster_modes.append('sequential')
            should_stop = False
        if not should_stop:
            print('Starting next stage')
    return {'selected_templates_max_pass_i': selected_templates_max_pass_i,
            'selected_templates_min_pass_i': selected_templates_min_pass_i, 'cluster_modes': cluster_modes,
            'should_stop': should_stop, }


def pursuit(self, op, data_json):
    out_dir = os.path.abspath(op['out_dir'])
    print('pursuit()', 'out_dir', out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if 'test' in op:
        if (('sample_num' in op['test']) and (op['test']['sample_num'] > 0) and (
                len(data_json) > op['test']['sample_num'])):
            print(('testing the procedure using a subsample of %d subtomograms' % op['test']['sample_num']))
            data_json = random.sample(data_json, op['test']['sample_num'])
    if 'segmentation' in op:
        if 'phi_propotion_cutoff' in op['segmentation']:
            raise Exception(
                'The phi_propotion_cutoff option has moved to template.guided_segmentation for better organization')
        segmentation_op = copy.deepcopy(op['segmentation'])
        segmentation_op['density_positive'] = op['density_positive']
        segmentation_tg_op = copy.deepcopy(segmentation_op)
        if 'guided_segmentation' in op['template']:
            if 'gaussian_smooth_sigma' in op['template']['guided_segmentation']:
                segmentation_tg_op['gaussian_smooth_sigma'] = op['template']['guided_segmentation'][
                    'gaussian_smooth_sigma']
            if 'phi_propotion_cutoff' in op['template']['guided_segmentation']:
                segmentation_tg_op['phi_propotion_cutoff'] = op['template']['guided_segmentation'][
                    'phi_propotion_cutoff']
    else:
        segmentation_op = None
        segmentation_tg_op = None
    data_json_file = os.path.join(out_dir, 'data.json')
    if os.path.exists(data_json_file):
        print(('loading ' + data_json_file))
        with open(data_json_file) as f:
            data_json = json.load(f)
    else:
        if op['random_initial_alignment']:
            print('force use random initial orientations')
        else:
            print('use initial transformations if avaliable')
        for rec in data_json:
            if ('loc' not in rec) or op['random_initial_alignment']:
                rec['loc'] = [0.0, 0.0, 0.0]
            if ('angle' not in rec) or op['random_initial_alignment']:
                rec['angle'] = [_ for _ in (N.random.random(3) * (N.pi * 2))]
        with open(data_json_file, 'wb') as f:
            json.dump(data_json, f, indent=2)
    v = IV.get_mrc(data_json[0]['subtomogram'])
    size = N.array(v.shape)
    del v
    file_stat_file = os.path.join(out_dir, 'file_stat.json')
    if os.path.isfile(file_stat_file):
        with open(file_stat_file) as f:
            file_stat = json.load(f)
        file_stat['passes'] = {int(_): file_stat['passes'][_] for _ in file_stat['passes']}
    else:
        file_stat = {'out_dir': out_dir,
                     'file_stat_file': file_stat_file,
                     'data_json_file': data_json_file,
                     'pass_i_current': 0,
                     'passes': {}}
        with open(file_stat_file, 'w') as f:
            json.dump(file_stat, f, indent=2)
    cluster_modes = ['kmeans']
    cluster_info = defaultdict(dict)
    cluster_info_stat = defaultdict(dict)
    cas_re = None
    fsc_stat = {}
    for pass_i in range(op['options']['pass_num']):
        if pass_i < file_stat['pass_i_current']:
            continue
        n_chunk = max(op['options']['min_chunk_size'],
                      int(N.ceil((len(data_json) / max(op['options']['min_worker_num'],
                                                       (self.runner.work_queue.get_worker_number() + 10))))))
        print('-------------------------------------------------------')
        print(('Beginning pass #%d' % pass_i))
        sys.stdout.flush()
        pass_start_time = time.time()
        pass_dir = os.path.join(out_dir, 'pass', ('%03d' % pass_i))
        if not os.path.exists(pass_dir):
            os.makedirs(pass_dir)
        data_json_file = os.path.join(pass_dir, 'data.json')
        cluster_info_file = os.path.join(pass_dir, 'cluster_info.pickle')
        cluster_info_stat_file = os.path.join(pass_dir, 'cluster_info_stat.pickle')
        cluster_modes_file = os.path.join(pass_dir, 'cluster_modes.json')
        cluster_average_select_file = os.path.join(pass_dir, 'cluster_average_select.pickle')
        align_template_file = os.path.join(pass_dir, 'align_template.pickle')
        fsc_stat_file = os.path.join(pass_dir, 'fsc_stat.json')
        if os.path.isfile(fsc_stat_file):
            print('loading', fsc_stat_file)
            with open(fsc_stat_file, 'r') as f:
                fsc_stat = json.load(f)
            fsc_stat = fsc_stat_json_convert(fsc_stat)
            if os.path.isfile(cluster_modes_file):
                print('loading', cluster_modes_file)
                with open(cluster_modes_file) as f:
                    cluster_modes = json.load(f)
            sts_re = stop_test_stat(op=op, fsc_stat=fsc_stat, pass_i=pass_i, cluster_modes=cluster_modes)
            cluster_modes = sts_re['cluster_modes']
            if sts_re['should_stop']:
                print('no more improvements seen, stop')
                break
            print('loading', data_json_file)
            with open(data_json_file) as f:
                data_json = json.load(f)
            print('loading ', cluster_average_select_file)
            with open(cluster_average_select_file, 'rb') as f:
                cas_re = pickle.load(f)
            print('loading', cluster_info_stat_file)
            with open(cluster_info_stat_file, 'rb') as f:
                cluster_info_stat = pickle.load(f)
            print('loading cluster info')
            for pass_i_t in file_stat['passes']:
                print('loading', file_stat['passes'][pass_i_t]['cluster_info_file'])
                with open(file_stat['passes'][pass_i_t]['cluster_info_file'], 'rb') as f:
                    cluster_info[pass_i_t] = pickle.load(f)
            if os.path.isfile(align_template_file):
                print(('loading ' + align_template_file))
                with open(align_template_file, 'rb') as f:
                    at_ress = pickle.load(f)
                at_ress = [_.result for _ in at_ress]
            print('....  and go to next pass')
            continue
        assert (pass_i not in file_stat['passes'])
        file_stat['passes'][pass_i] = {}
        file_stat['passes'][pass_i]['pass_i'] = pass_i
        file_stat['passes'][pass_i]['pass_dir'] = pass_dir
        file_stat['pass_i_current'] = pass_i
        op_file_t = os.path.join(pass_dir, ('pursuit-op-%d.json' % (int(time.time()),)))
        with open(op_file_t, 'w') as f:
            json.dump(op, f, indent=2)
        file_stat['passes'][pass_i]['cluster_modes_file'] = cluster_modes_file
        if os.path.isfile(cluster_modes_file):
            print('loading', cluster_modes_file)
            with open(cluster_modes_file) as f:
                cluster_modes = json.load(f)
        else:
            with open(cluster_modes_file, 'w') as f:
                json.dump(cluster_modes, f)
        data_json_dict = {_['subtomogram']: _ for _ in data_json}
        sys.stdout.flush()
        dimension_reduction_dump_file = os.path.join(pass_dir, 'dimension_reduction.pickle')
        file_stat['passes'][pass_i]['dimension_reduction_dump_file'] = dimension_reduction_dump_file
        if os.path.exists(dimension_reduction_dump_file):
            print(('loading ' + dimension_reduction_dump_file))
            with open(dimension_reduction_dump_file, 'rb') as f:
                dimension_reduction_dump_file__load = pickle.load(f)
            cfp_re = dimension_reduction_dump_file__load['cfp_re']
        else:
            start_time = time.time()
            data_json_pca_train = None
            if op['dim_reduction']['train_with_selected_clusters_only'] and (cas_re is not None):
                data_json_pca_train__set = set()
                data_json_pca_train = []
                for c in cas_re['selected_templates']:
                    t = cas_re['tk_info'][cas_re['selected_templates'][c]['subtomogram']]
                    if (op['dim_reduction']['restrict_to_specific_clusters'] and (
                            cluster_info_stat[t['pass_i']][t['cluster']]['is_specific'] is not None)):
                        continue
                    for d in t['data_json']:
                        if d['subtomogram'] in data_json_pca_train__set:
                            continue
                        data_json_pca_train__set.add(d['subtomogram'])
                        data_json_pca_train.append(data_json_dict[d['subtomogram']])
                del c, t
            if op['dim_reduction']['with_missing_wedge']:
                cfp_re = CU.covariance_filtered_pca_with_wedge(self=self, data_json=data_json, n_chunk=n_chunk, op=op,
                                                               pass_dir=pass_dir,
                                                               max_feature_num=op['dim_reduction']['max_feature_num'])
            else:
                cfp_re = CU.covariance_filtered_pca(self=self, data_json_model=data_json_pca_train,
                                                    data_json_embed=data_json,
                                                    normalize=op['dim_reduction']['normalize'], segmentation_tg_op=(
                        segmentation_tg_op if op['dim_reduction']['use_segmentation_mask'] else None), n_chunk=n_chunk,
                                                    pca_op=op['dim_reduction']['pca'],
                                                    max_feature_num=op['dim_reduction']['max_feature_num'])
            with open(dimension_reduction_dump_file, 'wb') as f:
                pickle.dump({'cfp_re': cfp_re, 'data_json_pca_train': data_json_pca_train, 'data_json': data_json, }, f,
                            protocol=(-1))
            print(('Dimension Reduction took time: %2.6f sec' % (time.time() - start_time)))
        sys.stdout.flush()
        cluster_dump_file = os.path.join(pass_dir, 'cluster.pickle')
        file_stat['passes'][pass_i]['cluster_dump_file'] = cluster_dump_file
        if os.path.exists(cluster_dump_file):
            print('loading', cluster_dump_file)
            with open(cluster_dump_file, 'rb') as f:
                cluster_dump_file__load = pickle.load(f)
            fsc_dict = cluster_dump_file__load['fsc_dict']
            clusters_populated = cluster_dump_file__load['clusters_populated']
        else:
            start_time = time.time()
            clusters_populated = {}
            fsc_dict = {}
            if 'kmeans' in set(cluster_modes):
                cluster_kmeans_dump_file = os.path.join(pass_dir, 'cluster__kmeans.pickle')
                file_stat['passes'][pass_i]['cluster_kmeans_dump_file'] = cluster_kmeans_dump_file
                if not os.path.isfile(cluster_kmeans_dump_file):
                    print('Generate subtomogram sets through kmeans clustering', end=' ')
                    kmeans_k = op['cluster']['kmeans']['number']
                    if 'kmeans-adaptave' not in set(cluster_modes):
                        kmeans_k = op['cluster']['kmeans']['number']
                    else:
                        specific_cluster_count = 0
                        for c in cas_re['selected_templates']:
                            t = cas_re['tk_info'][cas_re['selected_templates'][c]['subtomogram']]
                            if cluster_info_stat[t['pass_i']][t['cluster']]['is_specific'] is not None:
                                continue
                            specific_cluster_count += 1
                        kmeans_k = int(
                            N.round((specific_cluster_count * float(op['cluster']['kmeans']['adaptive_k_ratio']))))
                        del c, t, specific_cluster_count
                    print('k =', kmeans_k)
                    kmeans_labels = CU.kmeans_clustering(x=cfp_re['red'], k=kmeans_k)
                    kmeans_clusters = CU.labels_to_clusters(data_json=data_json, labels=kmeans_labels, cluster_mode=(
                        'kmeans' if ('kmeans-adaptave' not in set(cluster_modes)) else 'kmeans-adaptave'))
                    cluster_ssnr_fsc__op = {'ssnr': copy.deepcopy(op['ssnr'])}
                    if op['cluster']['ssnr']['segmentation']:
                        cluster_ssnr_fsc__op['segmentation_tg'] = copy.deepcopy(segmentation_tg_op)
                    cluster_ssnr_fsc = CU.cluster_ssnr_fsc(self=self,
                                                           clusters={_: kmeans_clusters[_]['data_json'] for _ in
                                                                     kmeans_clusters}, n_chunk=n_chunk,
                                                           op=cluster_ssnr_fsc__op)
                    with open(cluster_kmeans_dump_file, 'wb') as f:
                        pickle.dump(
                            {'cluster_ssnr_fsc': cluster_ssnr_fsc, 'kmeans_k': kmeans_k, 'kmeans_labels': kmeans_labels,
                             'kmeans_clusters': kmeans_clusters, }, f, protocol=(-1))
                else:
                    print('loading', cluster_kmeans_dump_file)
                    with open(cluster_kmeans_dump_file, 'rb') as f:
                        tmp = pickle.load(f)
                    cluster_ssnr_fsc = tmp['cluster_ssnr_fsc']
                    kmeans_labels = tmp['kmeans_labels']
                    kmeans_clusters = tmp['kmeans_clusters']
                    del tmp
                label_t = ((N.max([_ for _ in clusters_populated]) + 1) if (len(clusters_populated) > 0) else 0)
                for c in kmeans_clusters:
                    clusters_populated[label_t] = kmeans_clusters[c]
                    clusters_populated[label_t]['original_label'] = c
                    fsc_dict[label_t] = cluster_ssnr_fsc['fsc'][c]
                    label_t += 1
            if 'sequential' in set(cluster_modes):
                cluster_sequential_dump_file = os.path.join(pass_dir, 'cluster__sequential.pickle')
                file_stat['passes'][pass_i]['cluster_sequential_dump_file'] = cluster_sequential_dump_file
                if not os.path.isfile(cluster_sequential_dump_file):
                    print('Generate subtomogram sets through sequential expansion')
                    al_t = [_['align'] for _ in at_ress]
                    if (('filtering__second_largest_cut' in op['cluster']['sequential_expansion']) and (
                            len(al_t[0]) > 1)):
                        import aitom.tomominer.pursuit.multi.recursive.filtering.second_largest_cut as PMRFS
                        data_json_sp = PMRFS.do_filter(al=al_t, dj=data_json)
                    else:
                        data_json_sp = data_json
                    pmcra_op = copy.deepcopy(op['cluster']['sequential_expansion'])
                    pmcra_op['min_expansion_size'] = op['cluster']['size_min']
                    pmcra_op['n_chunk'] = n_chunk
                    pmcra_op['ssnr_sequential'] = {}
                    pmcra_op['ssnr_sequential']['ssnr'] = copy.deepcopy(op['ssnr'])
                    if op['cluster']['ssnr']['segmentation']:
                        pmcra_op['ssnr_sequential']['segmentation_tg'] = copy.deepcopy(segmentation_tg_op)
                    pmcra_re = CU.cluster_formation_alignment_fsc__by_global_maximum(self=self, dj=data_json_sp,
                                                                                     op=pmcra_op)
                    pmcra_re['data_json'] = data_json_sp
                    with open(cluster_sequential_dump_file, 'wb') as f:
                        pickle.dump(pmcra_re, f, protocol=(-1))
                else:
                    print('loading', cluster_sequential_dump_file)
                    with open(cluster_sequential_dump_file, 'rb') as f:
                        pmcra_re = pickle.load(f)
                label_t = ((N.max([_ for _ in clusters_populated]) + 1) if (len(clusters_populated) > 0) else 0)
                for c in pmcra_re['dj_gm']:
                    clusters_populated[label_t] = {'cluster_mode': 'sequential',
                                                   'data_json': pmcra_re['dj_gm'][c]['data_json'],
                                                   'original_label': c, }
                    fsc_dict[label_t] = pmcra_re['dj_gm'][c]['fsc']
                    label_t += 1
            assert (len(clusters_populated) > 0)
            with open(cluster_dump_file, 'wb') as f:
                pickle.dump({'fsc_dict': fsc_dict, 'clusters_populated': clusters_populated, }, f, protocol=(-1))
            print('Number of generated averages', len(clusters_populated))
            print(('Subtomogram average generation and FSC calculation  : %2.6f sec' % (time.time() - start_time)))
        for c in list(clusters_populated.keys()):
            if len(clusters_populated[c]['data_json']) < op['cluster']['size_min']:
                del clusters_populated[c]
        for c in clusters_populated:
            if c not in cluster_info[pass_i]:
                cluster_info[pass_i][c] = {}
            cluster_info[pass_i][c]['pass_i'] = pass_i
            cluster_info[pass_i][c]['cluster'] = c
            if c in fsc_dict:
                cluster_info[pass_i][c]['fsc'] = fsc_dict[c]
            cluster_info[pass_i][c]['cluster_mode'] = clusters_populated[c]['cluster_mode']
            cluster_info[pass_i][c]['data_json'] = clusters_populated[c]['data_json']
        for c in cluster_info[pass_i]:
            if pass_i not in cluster_info_stat:
                cluster_info_stat[pass_i] = {}
            if c not in cluster_info_stat[pass_i]:
                cluster_info_stat[pass_i][c] = {'cluster_size': len(cluster_info[pass_i][c]['data_json']), }
        sys.stdout.flush()
        cluster_averaging_file = os.path.join(pass_dir, 'cluster_averaging.pickle')
        file_stat['passes'][pass_i]['cluster_averaging_file'] = cluster_averaging_file
        if os.path.exists(cluster_averaging_file):
            print(('loading ' + cluster_averaging_file))
            with open(cluster_averaging_file, 'rb') as f:
                ca_re = pickle.load(f)
        else:
            print('Calculate subtomogram averages')
            cu_ca_op = copy.deepcopy(op['cluster']['averaging'])
            cu_ca_op['out_dir'] = os.path.join(pass_dir, 'clus_avg')
            cu_ca_op['pass_i'] = pass_i
            cu_ca_op['n_chunk'] = n_chunk
            if 'smooth' in cu_ca_op:
                cu_ca_op['smooth']['fsc'] = {_: fsc_dict[_] for _ in fsc_dict}
            ca_re = CU.cluster_averaging(self=self,
                                         clusters={_: clusters_populated[_]['data_json'] for _ in clusters_populated},
                                         op=cu_ca_op)
            with open(cluster_averaging_file, 'wb') as f:
                pickle.dump(ca_re, f, protocol=1)
        template_keys = ca_re['template_keys']
        for c in template_keys:
            if c not in cluster_info[pass_i]:
                cluster_info[pass_i][c] = {}
            cluster_info[pass_i][c]['template_key'] = template_keys[c]
            assert (cluster_info[pass_i][c]['template_key']['cluster'] == c)
            assert (cluster_info[pass_i][c]['template_key']['pass_i'] == pass_i)
        sys.stdout.flush()
        file_stat['passes'][pass_i]['cluster_average_select_file'] = cluster_average_select_file
        if os.path.exists(cluster_average_select_file):
            print(('loading ' + cluster_average_select_file))
            with open(cluster_average_select_file, 'rb') as f:
                cas_re = pickle.load(f)
        else:
            print('select averages')
            select_op = copy.deepcopy(op['cluster_average_select_fsc'])
            select_op['cluster'] = op['cluster']
            select_op['align_op'] = op['align']
            cas_re = CU.cluster_average_select_fsc(self=self, cluster_info=cluster_info,
                                                   cluster_info_stat=cluster_info_stat, op=select_op)
            if op['common_frame']['mode'] == 1:
                caacfmp = CU.cluster_average_align_common_frame__multi_pair(self=self, tk=cas_re['selected_templates'],
                                                                            align_op=op['align'], loc_r_max=(
                            size.min() * float(op['common_frame']['loc_r_max_proportion'])), pass_dir=pass_dir)
                cas_re['selected_templates_common_frame'] = caacfmp['tka']
                cas_re['selected_templates_common_frame__unrotated_clus'] = caacfmp['unrotated_clus']
                cas_re['selected_templates_common_frame__align_to_clus'] = caacfmp['align_to_clus']
                cas_re['cluster_average_align_common_frame__multi_pair_file'] = os.path.join(pass_dir,
                                                                                             'cluster_average_align_common_frame__multi_pair.pickle')
                with open(cas_re['cluster_average_align_common_frame__multi_pair_file'], 'wb') as f:
                    pickle.dump(caacfmp, f, protocol=(-1))
            elif op['common_frame']['mode'] == 2:
                caacfmp = CU.cluster_average_align_common_frame__single_best(self=self, tk=cas_re['selected_templates'],
                                                                             align_op=op['align'], pass_dir=pass_dir)
                cas_re['selected_templates_common_frame'] = caacfmp['tka']
                cas_re['cluster_average_align_common_frame__single_best_file'] = os.path.join(pass_dir,
                                                                                              'cluster_average_align_common_frame__single_best.pickle')
                with open(cas_re['cluster_average_align_common_frame__single_best_file'], 'wb') as f:
                    pickle.dump(caacfmp, f, protocol=(-1))
            else:
                raise Exception(("op['common_frame']['mode']" % (op['common_frame']['mode'],)))
            if segmentation_op is not None:
                print('segmenting the selected and aligned averages')
                template_segmentation_op = copy.deepcopy(segmentation_op)
                if ('segmentation' in op['template']) and (
                        'normalize_and_take_abs' in op['template']['segmentation']) and op['template']['segmentation'][
                    'normalize_and_take_abs']:
                    template_segmentation_op['normalize_and_take_abs'] = True
                CU.template_segmentation(self=self, tk=cas_re['selected_templates_common_frame'],
                                         op=template_segmentation_op)
            with open(cluster_average_select_file, 'wb') as f:
                pickle.dump(cas_re, f, protocol=(-1))
        if True:
            common_path_prefix = os.path.commonprefix(
                [cas_re['selected_templates'][_]['subtomogram'] for _ in cas_re['selected_templates']])
            print(len(cas_re['selected_templates']), 'averages selected.')
            print('Average list with common path prefix:', common_path_prefix)
            print('average id', '\t', 'generation mode', '\t', 'SFSC score', '\t', 'set size', '\t', 'average file')
            for _ in cas_re['selected_templates']:
                print(_, '\t', end=' ')
                print(cas_re['tk_info'][cas_re['selected_templates'][_]['subtomogram']]['cluster_mode'], '\t', end=' ')
                print(cas_re['tk_info'][cas_re['selected_templates'][_]['subtomogram']]['fsc'].sum(), '\t', end=' ')
                print(len(cas_re['tk_info'][cas_re['selected_templates'][_]['subtomogram']]['data_json']),
                      cas_re['selected_templates'][_]['subtomogram'][len(common_path_prefix):])
        sys.stdout.flush()
        file_stat['passes'][pass_i]['align_template_file'] = align_template_file
        if os.path.isfile(align_template_file):
            print(('loading ' + align_template_file))
            with open(align_template_file, 'rb') as f:
                at_ress = pickle.load(f)
        else:
            align_template__tmp_dir__file = os.path.join(pass_dir, 'align_template__tmp_dir.json')
            if os.path.isfile(align_template__tmp_dir__file):
                with open(align_template__tmp_dir__file) as f:
                    align_template__tmp_dir = json.load(f)
            else:
                align_template__tmp_dir = os.path.join(self.cache.tmp_dir, ('align-template-' + str(uuid.uuid4())))
                with open(align_template__tmp_dir__file, 'w') as f:
                    json.dump(align_template__tmp_dir, f, indent=2)
            start_time = time.time()
            if not os.path.isdir(align_template__tmp_dir):
                os.makedirs(align_template__tmp_dir)
            at_ress = CU.align_to_templates__batch(self=self, op=op, data_json=data_json,
                                                   segmentation_tg_op=segmentation_tg_op,
                                                   tmp_dir=align_template__tmp_dir,
                                                   tem_keys=cas_re['selected_templates_common_frame'])
            with open(align_template_file, 'wb') as f:
                pickle.dump(at_ress, f, protocol=(-1))
            shutil.rmtree(align_template__tmp_dir)
            print(('Align all volumes to cluster_templates. %2.6f sec' % (time.time() - start_time)))
        at_ress = [_.result for _ in at_ress]
        sys.stdout.flush()
        cratcms = CU.cluster_removal_according_to_center_matching_specificity(ci=cluster_info, cis=cluster_info_stat,
                                                                              al=at_ress,
                                                                              tk=cas_re['selected_templates'],
                                                                              significance_level=op[
                                                                                  'cluster_removal_according_to_center_matching_specificity'][
                                                                                  'significance_level'])
        with open(os.path.join(pass_dir, 'cluster_removal_according_to_center_matching_specificity.pickle'), 'wb') as f:
            pickle.dump(cratcms, f, protocol=(-1))
        with open(cluster_info_file, 'wb') as f:
            pickle.dump(cluster_info[pass_i], f, protocol=(-1))
        file_stat['passes'][pass_i]['cluster_info_file'] = cluster_info_file
        with open(cluster_info_stat_file, 'wb') as f:
            pickle.dump(cluster_info_stat, f, protocol=(-1))
        file_stat['passes'][pass_i]['cluster_info_stat_file'] = cluster_info_stat_file
        data_json_new = []
        for res in at_ress:
            rec = {'subtomogram': res['vol_key']['subtomogram'],
                   'mask': res['vol_key']['mask'],
                   'angle': [_ for _ in res['best']['angle']],
                   'loc': [_ for _ in res['best']['loc']],
                   'score': res['best']['score']}
            if res['best']['template_id'] is not None:
                rec['template'] = cas_re['selected_templates_common_frame'][res['best']['template_id']]
                assert ('id' in rec['template'])
            data_json_new.append(rec)
        data_json = data_json_new
        with open(data_json_file, 'w') as f:
            json.dump(data_json, f, indent=2)
        file_stat['passes'][pass_i]['data_json_file'] = data_json_file
        file_stat['passes'][pass_i]['fsc_stat_file'] = fsc_stat_file
        if os.path.exists(fsc_stat_file):
            print(('loading' + fsc_stat_file))
            with open(fsc_stat_file) as f:
                fsc_stat = json.load(f)
            fsc_stat = fsc_stat_json_convert(fsc_stat)
        else:
            fsc_stat_t = {}
            for _ in cas_re['selected_templates']:
                kt = cas_re['selected_templates'][_]['subtomogram']
                fsc_stat_t[_] = {'pass_i': cas_re['tk_info'][kt]['pass_i'], 'cluster': cas_re['tk_info'][kt]['cluster'],
                                 'fsc': cas_re['tk_info'][kt]['fsc'].sum(), }
                fsc_stat_t[_]['cluster_mode'] = cluster_info[fsc_stat_t[_]['pass_i']][fsc_stat_t[_]['cluster']][
                    'cluster_mode']
            fsc_stat[pass_i] = fsc_stat_t
            del fsc_stat_t, kt, _
            for (pass_i_t, fsc_stat_t) in fsc_stat.items():
                for (template_i, fsc_stat_tt) in fsc_stat_t.items():
                    fsc_stat_tt['is_specific'] = cluster_info_stat[fsc_stat_tt['pass_i']][fsc_stat_tt['cluster']][
                        'is_specific']
            del pass_i_t, fsc_stat_t, template_i, fsc_stat_tt
            with open(fsc_stat_file, 'w') as f:
                json.dump(fsc_stat, f, indent=2)
        with open(file_stat_file, 'w') as f:
            json.dump(file_stat, f, indent=2)
        sts_re = stop_test_stat(op=op, fsc_stat=fsc_stat, pass_i=pass_i, cluster_modes=cluster_modes)
        cluster_modes = sts_re['cluster_modes']
        if sts_re['should_stop']:
            print('no more improvements seen, stop')
            break
        print(('Entire pass: %2.6f sec' % (time.time() - pass_start_time)))
        sys.stdout.flush()
    return file_stat

import os
import json
import math
from tqdm import tqdm

import torch
import torchnet as tnt

from .protonets.utils import filter_opt, merge_dict
from .protonets.utils import data as data_utils
from .protonets.utils import model as model_utils


def main(opt):
    # load model
    model = torch.load(opt['model.model_path'])
    model.eval()

    # load opts
    model_opt_file = os.path.join(os.path.dirname(opt['model.model_path']), 'opt.json')
    with open(model_opt_file, 'r') as f:
        model_opt = json.load(f)

    # Postprocess arguments
    model_opt['model.x_dim'] = map(int, model_opt['model.x_dim'].split(','))
    model_opt['log.fields'] = model_opt['log.fields'].split(',')

    # construct data
    data_opt = {'data.' + k: v for k, v in filter_opt(model_opt, 'data').items()}

    episode_fields = {
        'data.test_way': 'data.way',
        'data.test_shot': 'data.shot',
        'data.test_query': 'data.query',
        'data.test_episodes': 'data.train_episodes'
    }

    for k, v in episode_fields.items():
        if opt[k] != 0:
            data_opt[k] = opt[k]
        elif model_opt[k] != 0:
            data_opt[k] = model_opt[k]
        else:
            data_opt[k] = model_opt[v]

    print("Evaluating {:d}-way, {:d}-shot with {:d} query examples/class over {:d} episodes".format(
        data_opt['data.test_way'], data_opt['data.test_shot'],
        data_opt['data.test_query'], data_opt['data.test_episodes']))

    torch.manual_seed(1234)
    if data_opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    data = data_utils.load(data_opt, ['test'])

    if data_opt['data.cuda']:
        model.cuda()

    meters = {field: tnt.meter.AverageValueMeter() for field in model_opt['log.fields']}
    if opt['stage'] == 'protonet':
        _, class_acc, class_prec, prec_micro = model_utils.evaluate(model,
                                                                    data['test'], meters, stage='protonet',
                                                                    desc="test", evaluation=True)
    else:
        _, class_acc, class_prec, prec_micro = model_utils.evaluate(model,
                                                                    data['test'], meters, stage='feat',
                                                                    desc='test', evaluation=True)
    for field, meter in meters.items():
        mean, std = meter.value()
        print("test {:s}: {:0.6f} +/- {:0.6f}".format(field, mean,
                                                      1.96 * std / math.sqrt(data_opt['data.test_episodes'])))

    mean_prec = 0
    n = 0
    for k in class_acc.keys():
        print('class {} acc: {:0.4f}'.format(k, class_acc[k]))
    for k in class_prec.keys():
        mean_prec += class_prec[k]
        n += 1
        print('class {} prec: {:0.4f}'.format(k, class_prec[k]))
    mean_prec = mean_prec / n
    print('Average prec(macro): {:0.4f}; Average prec(micro): {:0.4f}'.format(mean_prec, prec_micro))

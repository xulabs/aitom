from tqdm import tqdm

from . import filter_opt
from ..models import get_model


def load(opt):
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)


def evaluate(model, data_loader, meters, stage, desc=None, evaluation=False):
    model.eval()

    for field, meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    class_acc_all = {}
    class_count_all = {}
    class_prec_all = {}
    class_prec_count_all = {}

    for sample in data_loader:
        if stage == 'protonet':
            _, output, class_acc, class_count, class_prec = model.loss(sample, stage, eval=evaluation)
        elif stage == 'feat':
            _, output, _, class_acc, class_count, class_prec = model.loss(sample, stage, eval=evaluation)
        for k in class_acc.keys():
            if not k in class_acc_all.keys():
                class_count_all[k] = 1
                class_acc_all[k] = class_acc[k]
            else:
                class_count_all[k] += 1
                class_acc_all[k] += class_acc[k]
        for k in class_prec.keys():
            if not k in class_prec_all.keys():
                class_prec_count_all[k] = class_count[k]
                class_prec_all[k] = class_prec[k]
            else:
                class_prec_count_all[k] += class_count[k]
                class_prec_all[k] += class_prec[k]
        for field, meter in meters.items():
            meter.add(output[field])
    for k in class_count_all.keys():
        class_acc_all[k] = class_acc_all[k] / class_count_all[k]
    prec_micro = 0
    count_micro = 0
    for k in class_prec_all.keys():
        prec_micro += class_prec_all[k]
        count_micro += class_prec_count_all[k]
        class_prec_all[k] = class_prec_all[k] / class_prec_count_all[k]
    prec_micro = prec_micro / count_micro

    return meters, class_acc_all, class_prec_all, prec_micro

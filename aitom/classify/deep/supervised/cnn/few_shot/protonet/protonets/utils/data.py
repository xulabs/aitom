import protonets.data


def load(opt, splits):
    if opt['data.dataset'] == 'subtomo':
        ds = protonets.data.subtomo.load(opt, splits)
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds

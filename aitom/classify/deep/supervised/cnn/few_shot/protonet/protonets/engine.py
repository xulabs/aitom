from tqdm import tqdm
import torch
import torch.nn.functional as F


class Engine(object):
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = {}
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'stage': kwargs['stage'],
            'epoch': 0,  # epochs done so far
            't': 0,  # samples seen so far
            'batch': 0,  # samples seen in current epoch
            'stop': False
        }
        args = kwargs['args']
        print(args)
        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])
        # construct attention label
        att_label_basis = []
        for i in range(args['data.way']):
            temp = torch.eye(args['data.way'] + 1)
            temp[i, i] = 0.5
            temp[-1, -1] = 0.5
            temp[i, -1] = 0.5
            temp[-1, i] = 0.5
            att_label_basis.append(temp)

        label = torch.arange(args['data.way'], dtype=torch.int8).repeat(args['data.query'])
        att_label = torch.zeros(label.shape[0], args['data.way'] + 1, args['data.way'] + 1)
        for i in range(att_label.shape[0]):
            att_label[i, :] = att_label_basis[label[i].item()]
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
            att_label = att_label.cuda()

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:
            state['model'].train()

            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):
                state['sample'] = sample
                self.hooks['on_sample'](state)

                state['optimizer'].zero_grad()
                if state['stage'] == 'protonet':
                    loss, state['output'], _, _, _ = state['model'].loss(state['sample'], state['stage'])
                    self.hooks['on_forward'](state)
                elif state['stage'] == 'feat':
                    loss, state['output'], att, _, _, _ = state['model'].loss(state['sample'], state['stage'])
                    self.hooks['on_forward'](state)
                    loss_att = F.kl_div(att.view(-1, args['data.way'] + 1), att_label.view(-1, args['data.way'] + 1))
                    loss = loss + 2 * loss_att
                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)

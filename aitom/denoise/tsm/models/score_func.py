import torch.nn as nn
from models.networks import build_network

class ScoreFuncModel(nn.Module):
    def __init__(self, config):
        
        super().__init__()
        self.network = build_network(config.network_config)

        self.additional_sigmas = config.additional_sigmas
        self.milestones = config.milestones
        self.milestones_intervals = config.milestones_intervals

        self.count = 0
        self.additional_sigma = self.additional_sigmas[0]


    def forward(self, batch, mode='test'):
        self._update_sigma()
        x = batch["noiser_images"]
        if mode=='train':
            out = self.network(x) * self.additional_sigma
        else:
            out = self.network(x)
        return out

    def _update_sigma(self):

        # no milestones or only one sigma: just count
        if len(self.milestones) <= 1 or len(self.additional_sigmas) <= 1: 
            self.count += 1
            return
        
        # use sliding window to update milestones and sigmas
        if self.count > self.milestones[1]:
            self.milestones = self.milestones[1:]
            self.additional_sigmas = self.additional_sigmas[1:]
            self.milestones_intervals = self.milestones_intervals[1:]

        # after sliding, if less than 2 points remain, just count 
        if len(self.milestones) <= 1 or len(self.additional_sigmas) <= 1:
            self.count += 1
            return

        # linearly interpolate additional_sigma between milestones
        start, end = self.milestones[0], self.milestones[1]
        interval = self.milestones_intervals[0]
        denom = max(1, end - start)

        stepped = ((self.count - start) // interval) * interval  # step by interval
        fraction = stepped / denom

        s0, s1 = self.additional_sigmas[0], self.additional_sigmas[1]
        self.additional_sigma = s0 + fraction * (s1 - s0)

        self.count += 1

    

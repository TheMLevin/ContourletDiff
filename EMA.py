# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

'''
Codes adapted from https://github.com/NVlabs/LSGM/blob/main/util/ema.py
'''
import warnings

import torch
from torch.optim import Optimizer


class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        # stop here if we are not applying EMA
        if not self.apply_ema:
            return retval

        ema, params = {}, {}
        for group in self.optimizer.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]

                # State initialization
                if 'ema' not in state:
                    state['ema'] = p.data.clone()

                if p.shape not in params:
                    params[p.shape] = {'idx': 0, 'data': []}
                    ema[p.shape] = []

                params[p.shape]['data'].append(p.data)
                ema[p.shape].append(state['ema'])

            for i in params:
                params[i]['data'] = torch.stack(params[i]['data'], dim=0)
                ema[i] = torch.stack(ema[i], dim=0)
                ema[i].mul_(self.ema_decay).add_(params[i]['data'], alpha=1. - self.ema_decay)

            for p in group['params']:
                if p.grad is None:
                    continue
                idx = params[p.shape]['idx']
                self.optimizer.state[p]['ema'] = ema[p.shape][idx, :]
                params[p.shape]['idx'] += 1

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        # load_state_dict loads the data to self.state and self.param_groups. We need to pass this data to
        # the underlying optimizer too.
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema=True):
        """
        Swap current parameters with their EMA versions.

        If a parameter does not have an 'ema' entry (e.g., it never received
        a gradient), we skip it safely.
        """

        if not self.apply_ema:
            warnings.warn('swap_parameters_with_ema was called but EMA is disabled.')
            return

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue

                state = self.optimizer.state[p]
                ema = state.get('ema', None)

                # If EMA for this parameter does not exist, skip it.
                if ema is None:
                    continue

                # Swap logic
                if store_params_in_ema:
                    tmp = p.data.clone()
                    p.data.copy_(ema)
                    state['ema'] = tmp
                else:
                    p.data.copy_(ema)

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import torch

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()

import torch
torch.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(0)

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()

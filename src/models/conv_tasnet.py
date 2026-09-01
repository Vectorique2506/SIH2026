"""
Conv-TasNet model — alternative to DCCRN.

Owner: Track 2 (Model)

Pick ONE of dccrn.py / conv_tasnet.py to actually pursue. Keep the other
as a backup only if you have spare time near the end.

Reference to clone from:
    git clone https://github.com/kaituoxu/Conv-TasNet.git /tmp/convtasnet_ref
"""

import torch
import torch.nn as nn


class ConvTasNet(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: paste/adapt architecture from cloned reference
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

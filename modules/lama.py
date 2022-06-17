import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter


class LAMA(Module):
    """
    This class implements the low rank factorization multi-head self-attention mechanism described
    in `"Low Rank Factorization for Compact Multi-Head Self-Attention"
    <https://arxiv.org/abs/1912.00835>`_ by Mehta et al., 2019.

    Inputs:

    - inputs: shape ``(batch_size, max_sequence_length, input_dim)``
    - mask: shape ``(batch_size, max_sequence_length)``, should be 0 at timesteps where att
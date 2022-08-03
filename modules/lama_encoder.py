from typing import Callable

import torch
from torch.nn import Linear
from torch.nn import Module

from .lama import LAMA


class LAMAEncoder(Module):
    """
    This class implements the encoder proposed in `"Low Rank Factorization for Compact Multi-Head Self-Attention"
    <https://arxiv.org/abs/1912.00835>`_ by Mehta et al., 2019. This applies the attention weights of the LAMA
    attention mechanism to the inputs returning the structured sentence embedding.

    Inputs:

    - inputs: 
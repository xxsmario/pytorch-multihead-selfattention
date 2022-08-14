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

    - inputs: shape ``(batch_size, max_sequence_length, input_dim)``
    - mask: shape ``(batch_size, max_sequence_length)``, should be 0 at timesteps where attention will be masked
        (e.g. pad tokens), and 1 otherwise.

    Output:

    If ``output_dim`` is not None:
        - structured sentence embedding: shape ``(batch_size, num_heads, input_dim)``.
    Else:
        - document embedding: shape ``(batch_size, output_dim)``.


    Parameters
    ----------
    num_heads : ``int``, required.
        The number of attention heads to use.
    input_dim : ``int``, required.
        The size of the last dimension of the input tensor.
    activation : ``Callable``, optional (default=``torch.tanh``)
        An activation function applied after the attention calculation. Default is
        ``torch.tanh``. Set to ``None`` to use linear activation (i.e. no activation).
    output_dim : ``Optional[int]``, optional (default=``None``)
        If not None, we'll apply the computed attention weights for each head to ``inputs``, concatenate the
        resulting features, and project them into a vector of this size, giving an output of
        ``(batch_size, output_dim)``. If this value is ``None``, we will just return the attention
import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parameter import Parameter


class LAMA(Module):
    """
    This
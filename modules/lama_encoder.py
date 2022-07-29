from typing import Callable

import torch
from torch.nn import Linear
from torch.nn import Module

from .lama import LAMA


class LAMAEncoder(Modul
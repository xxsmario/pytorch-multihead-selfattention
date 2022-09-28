import pytest
import torch

from modules.lama import LAMA
from modules.lama_encoder import LAMAEncoder


@pytest.fixture
def lama():
    """Return a tuple of the args used to intialize ``LAMA`` and the initialized instance.
    """
    # This nested function lets us build the object on the fly in our unit tests
    def _initialize(num_heads=6, input_dim=128, activation=torch.ta
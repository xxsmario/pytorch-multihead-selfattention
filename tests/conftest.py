import pytest
import torch

from modules.lama import LAMA
from modules.lama_encoder import LAMAEncoder


@pytest.fixture
def lama():
    """Return a tuple of the a
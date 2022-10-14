import torch


class TestLAMA(object):
    """Collects all unit tests for `modules.lama.LAMA`.
    """
    def test_attributes_after_initialization(self, lama):
        args, lama = lama()

        assert lama._activation == args['activati
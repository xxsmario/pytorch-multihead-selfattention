import torch


class TestLAMA(object):
    """Collects all unit tests for `modules.lama.LAMA`.
    """
    def test_attributes_after_initialization(self, lama):
        args, lama = lama()

        assert lama._activation == args['activation']
        assert lama._normalize == args['normalize']
        assert lama._p.size() == (args['input_dim'], args['num_heads'])
   
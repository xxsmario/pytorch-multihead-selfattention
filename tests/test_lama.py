import torch


class TestLAMA(object):
    """Collects all unit tests for `modules.lama.LAMA`.
    """
    def test_attributes_after_initialization(self, lama):
        args, lama = lama()

        assert lama._activation == args['activation']
        assert lama._normalize == args['normalize']
        assert lama._p.size() == (args['input_dim'], args['num_heads'])
        assert lama._q.size() == (args['input_dim'], args['num_heads'])

    def test_output_shape_forward_without_mask_without_normalization(self, lama):
        args, lama = lama(normalize=False)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        output = lama(inputs)

        assert output.size() == (batch_size, args['num_heads'], max_seq_len)

    def test_output_shape_forward_without_mask_with_normalization(self, lama):
      
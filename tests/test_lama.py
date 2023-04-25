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
        args, lama = lama(normalize=True)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        output = lama(inputs)

        assert output.size() == (batch_size, args['num_heads'], max_seq_len)

    def test_output_shape_forward_with_mask_without_normalization(self, lama):
        args, lama = lama(normalize=False)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = torch.ones(batch_size, max_seq_len)
        mask[:, -1] = 0  # Zero-out the last timestep of each sequence
        output = lama(inputs, mask)

        assert output.size() == (batch_size, args['num_heads'], max_seq_len)

    def test_output_shape_forward_with_mask_with_normalization(self, lama):
        args, lama = lama(normalize=True)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = torch.ones(batch_size, max_seq_len)
        mask[:, -1] = 0  # Zero-out the last timestep of each sequence
        output = lama(inputs, mask)

        assert output.size() == (batch_size, args['num_heads'], max_seq_len)

    def test_output_is_masked_forward(self, lama):
        args, lama = lama(normalize=True)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = torch.ones(batch_size, max_seq_len)
        mask[:, -1] = 0  # Zero-out the last timestep of each sequence
        output = lama(inputs, mask)

        # Make sure the probability is 0 at masked positions
        assert torch.allclose(output[:, :, -1], torch.zeros_like(output[:, :, -1]))

    def test_output_shape_forward_internal_without_mask(self, lama):
        args, lama = lama()

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = None
        output = lama._forward_internal(inputs, mask)

        assert output.size() == (batch_size, args['num_heads'], max_seq_len)

    def test_output_shape_forward_internal_with_mask(self, lama):
        args, lama = lama()

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = torch.ones(batch_size, max_seq_len)
        mask[:, -1] = 0  # Zero-out the last timestep of each sequence
        output = lama._forward_internal(inputs, mask)

        assert output.size() == (batch_size, args['num_heads'], max_seq_len)

    def test_output_sums_to_one_forward_without_mask(self, lama):
        args, lama = lama(normalize=True)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        output = lama(inputs)

        # Attention weights should sum to 1
        assert torch.allclose(torch.sum(output, dim=-1), torch.ones(batch_size, args['num_heads']))

    def test_output_sums_to_one_forward_with_mask(self, lama):
        args, lama = lama(normalize=True)

        # Keep these small so testing is fast
        batch_size = 4
        max_seq_len = 25  # Maximum length of the input sequence

        inputs = torch.randn(batch_size, max_seq_len, args['input_dim'])
        mask = torch.ones(batch_size, max_seq_len)
        mask[:, -1] = 0  # Zero-out the last timestep of each sequence
        output = lama(inputs, mask)

        # Attention weights should sum to 1
        assert torch.allclose(torch.sum(output, dim=-1), torch.ones(batch_size, args['num_heads']))

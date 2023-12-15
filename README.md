
[![Build Status](https://travis-ci.org/xxsmario/pytorch-multihead-selfattention.svg?branch=master)](https://travis-ci.org/xxsmario/pytorch-multihead-selfattention)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/910dc1e31a6a4e0daff8ae42b7d3205f)](https://www.codacy.com/manual/xxsmario/pytorch-multihead-selfattention?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=xxsmario/pytorch-multihead-selfattention&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/910dc1e31a6a4e0daff8ae42b7d3205f)](https://www.codacy.com/manual/xxsmario/pytorch-multihead-selfattention?utm_source=github.com&utm_medium=referral&utm_content=xxsmario/pytorch-multihead-selfattention&utm_campaign=Badge_Coverage)

# PyTorch Based Multi-Head Self-Attention Mechanism - LAMA

This is an advanced PyTorch library that implements the __L__ow Rank F__a__ctorization for Compact __M__ulti-Head __A__ttention (LAMA) mechanism and the corresponding pooler as introduced in the paper: "[Low Rank Factorization for Compact Multi-Head Self-Attention](https://arxiv.org/abs/1912.00835)".

![](img/figure_1.jpg)

> Figure 1 from "Low Rank Factorization for Compact Multi-Head Self-Attention".

## Usage

PyTorch is the only dependency. Installation instructions can be found [here](https://pytorch.org/get-started/locally/).

Detailed instructions for the usage of LAMA and LAMAEncoder are provided.

```python
import torch
from modules.lama import LAMA

num_heads = 8      # Number of attention heads
input_dim = 768    # Dimension of each tokens hidden representation
batch_size = 16    # Number of sentences/documents in the mini-batch
max_seq_len = 100  # Maximum length of the input sequence

# Create a random input sequence
inputs = torch.randn(batch_size, max_seq_len, input_dim)  
# Optionally, you can provide a mask over timesteps (e.g., for padding tokens)
# Size: (batch_size, max_seq_len), 0 where timesteps should be masked and 1 otherwise
mask = torch.ones(batch_size, max_seq_len)
mask[:, -1] = 0

# Initialize the attention mechanism
lama = LAMA(num_heads, input_dim)
output = lama(inputs, mask)

assert output.size() == (batch_size, num_heads, max_seq_len)
```

```python
import torch
from modules.lama_encoder import LAMAEncoder

num_heads = 8      # Number of attention heads
input_dim = 768    # Dimension of each tokens hidden representation
batch_size = 16    # Number of sentences/documents in the mini-batch
max_seq_len = 100  # Maximum length of the input sequence

# Create a random input sequence
inputs = torch.randn(batch_size, max_seq_len, input_dim)  
# Optionally, you can provide a mask over timesteps (e.g., for padding tokens)
# Size: (batch_size, max_seq_len), 0 where timesteps should be masked and 1 otherwise
mask = torch.ones(batch_size, max_seq_len)
mask[:, -1] = 0

# Initialize the encoder
lama_encoder = LAMAEncoder(num_heads, input_dim)
output = lama_encoder(inputs, mask)

assert output.size() == (batch_size, num_heads, input_dim)

# If output_dim is not None (default), the "structured sentence embedding" is flattened by concatenation and projected by a linear layer into a vector of this size
lama_encoder = LAMAEncoder(num_heads, input_dim, output_dim=128)
output = lama_encoder(inputs, mask)

assert output.size() == (batch_size, 128)
```
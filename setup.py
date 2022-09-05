import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lama",
    version="0.1.0",
    author="John Giorgi",
    author_email="johnmgiorgi@gmail.com",
    description=("A PyTorch implementation of the Compact Multi-Head Self-Attention Mechanism from"
                 " the paper: 'Low Rank Factorization for Compact Multi-Head Self-Attention'"
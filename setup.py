from setuptools import find_packages, setup

setup(
    name='drg_tools',
    version='0.0.1',
    author='Alexander Sasse',
    author_email='alexander.sasse@gmail.com',
    packages=find_packages(),
    license='LICENSE',
    description='drg_tools contains classes and functions to create and analyze sequence-to-function models.',
    install_requires=[
        "numpy >= 1.14.2",
        "torch >= 1.9.0",
        "fft_conv_pytorch == 1.2.0",
        "einops == 0.8.0",
    ],
)

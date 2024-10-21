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
        "einops == 0.8.0",
        "fft_conv_pytorch == 1.2.0",
        "joblib",
        "logomaker>=0.8",
        "matplotlib>=3.8",
        "numpy >= 1.26",
        "pandas>=2.2.2",
        "scikit-learn>=1.4.2",
        "scipy>=1.13",
        "seaborn>=0.13.2",
        "torch >= 2.3.1",
    ],
)

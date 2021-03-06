""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

__version__ = '0.0.2'
setup(
    name='hdvw',
    version=__version__,
    description='PyTorch Image Models',
    long_description_content_type='text/markdown',
    url='https://github.com/shaoshitong/hdvw.git',
    author='sst',
    author_email='10907843@qq.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch models efficientnet mobilenetv3 mnasnet',
    packages=find_packages(exclude=['checkpoints','configs','models','ops','resources']),
    package_dir={'hdvw':'hdvw'},
    install_requires=['torch >= 1.4', 'torchvision'],
    include_package_data=True,
    python_requires='>=3.6',
)

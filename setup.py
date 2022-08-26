from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='prnn_cpp',
    ext_modules=[
        CppExtension('prnn_cpp', ['prnn.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

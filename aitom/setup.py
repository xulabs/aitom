#!/usr/bin/env python


import os
import platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

compile_extra_args = ['-std=c++11']
link_extra_args = []
if platform.system() == "Darwin":
    compile_extra_args = ['-std=c++11', "-mmacosx-version-min=10.9"]
    link_extra_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

cpp_core = Extension('tomominer.core.core',
                     sources=[
                         'tomominer/core/cython/wrap_core.cpp',
                         'tomominer/core/cython/core.pyx',
                         'tomominer/core/src/affine_transform.cpp',
                         'tomominer/core/src/align.cpp',
                         'tomominer/core/src/arma_extend.cpp',
                         'tomominer/core/src/dilate.cpp',
                         'tomominer/core/src/fft.cpp',
                         'tomominer/core/src/geometry.cpp',
                         'tomominer/core/src/interpolation.cpp',
                         'tomominer/core/src/io.cpp',
                         'tomominer/core/src/legendre.cpp',
                         'tomominer/core/src/rotate.cpp',
                         'tomominer/core/src/sht.cpp',
                         'tomominer/core/src/wigner.cpp',
                         'tomominer/core/src/segmentation/watershed/watershed_segmentation.cpp'],
                     libraries=['m', 'fftw3', 'armadillo', 'blas', 'lapack'],
                     include_dirs=[get_include(), '/usr/include',
                                   '/usr/local/include', 'tomominer/core/src/'],
                     library_dirs=[],
                     extra_compile_args=compile_extra_args,
                     extra_link_args=link_extra_args,
                     language='c++')


def get_packages(root_dir='tomominer', exclude_dir_roots=['tomominer/core/src', 'tomominer/core/cython']):
    pkg = []
    for (root, dirs, files) in os.walk(root_dir):
        exclude = False
        for d in exclude_dir_roots:
            if root.startswith(d):
                exclude = True
        if exclude:
            continue
        pkg.append(root.replace('/', '.'))
    return pkg


setup(name='tomominer',
      version='0.9.0',
      author='Alber Lab (USC)',
      description='Subtomogram Analysis and Mining Software',
      license='GPLv3',
      url='',
      platforms=['x86_64'],
      ext_modules=[cpp_core],
      packages=get_packages(),
      package_dir={'tomominer': 'tomominer',
                   'tomominer.core': 'tomominer/core/cython/', },
      cmdclass={'build_ext': build_ext, })

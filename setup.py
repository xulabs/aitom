#!/usr/bin/env python


import os
import platform
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from numpy import get_include

compile_extra_args = ['-std=c++11']
link_extra_args = []
if platform.system() == "Darwin":
    compile_extra_args = ['-std=c++11', "-mmacosx-version-min=10.9"]
    link_extra_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

cpp_core = Extension('aitom.tomominer.core.core',
                     sources=[
                         'aitom/tomominer/core/cython/wrap_core.cpp',
                         'aitom/tomominer/core/cython/core.pyx',
                         'aitom/tomominer/core/src/affine_transform.cpp',
                         'aitom/tomominer/core/src/align.cpp',
                         'aitom/tomominer/core/src/arma_extend.cpp',
                         'aitom/tomominer/core/src/dilate.cpp',
                         'aitom/tomominer/core/src/fft.cpp',
                         'aitom/tomominer/core/src/geometry.cpp',
                         'aitom/tomominer/core/src/interpolation.cpp',
                         'aitom/tomominer/core/src/io.cpp',
                         'aitom/tomominer/core/src/legendre.cpp',
                         'aitom/tomominer/core/src/rotate.cpp',
                         'aitom/tomominer/core/src/sht.cpp',
                         'aitom/tomominer/core/src/wigner.cpp',
                         'aitom/tomominer/core/src/segmentation/watershed/watershed_segmentation.cpp'],
                     libraries=['m', 'fftw3', 'armadillo', 'blas', 'lapack'],
                     include_dirs=[get_include(), '/usr/include',
                                   '/usr/local/include', 'aitom/tomominer/core/src/'],
                     library_dirs=[],
                     extra_compile_args=compile_extra_args,
                     extra_link_args=link_extra_args,
                     language='c++')


def get_packages(root_dir='aitom', exclude_dir_roots=['aitom/tomominer/core/src', 'aitom/tomominer/core/cython']):
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


setup(name='aitom',
      version='0.0.1',
      author='Xu Lab (CMU) and collaborators',
      author_email='mxu1@cs.cmu.edu',
      description='AI software for tomogram analysis',
      license='GPLv3',
      url='https://github.com/xulabs/aitom',
      platforms=['x86_64'],
      ext_modules=cythonize(cpp_core),
      packages=get_packages(),
      package_dir={'aitom': 'aitom',
                   'aitom.tomominer.core': 'aitom/tomominer/core/cython/', },
      cmdclass={'build_ext': build_ext, })

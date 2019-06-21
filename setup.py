#!/usr/bin/env python



import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
cpp_core = Extension('tomominer.core.core', sources=['tomominer/core/cython/wrap_core.cpp', 'tomominer/core/cython/core.pyx', 'tomominer/core/src/affine_transform.cpp', 'tomominer/core/src/align.cpp', 'tomominer/core/src/arma_extend.cpp', 'tomominer/core/src/dilate.cpp', 'tomominer/core/src/fft.cpp', 'tomominer/core/src/geometry.cpp', 'tomominer/core/src/interpolation.cpp', 'tomominer/core/src/io.cpp', 'tomominer/core/src/legendre.cpp', 'tomominer/core/src/rotate.cpp', 'tomominer/core/src/sht.cpp', 'tomominer/core/src/wigner.cpp', 'tomominer/core/src/segmentation/watershed/watershed_segmentation.cpp'], libraries=['m', 'fftw3', 'armadillo', 'blas', 'lapack'], include_dirs=[get_include(), '/usr/include', 'tomominer/core/src/'], library_dirs=[os.path.join(os.getenv('HOME'), 'local/lib64/')], extra_compile_args=['-std=c++11'], language='c++')

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
setup(name='tomominer', version='0.9.0', author='Alber Lab (USC)', description='Subtomogram Analysis and Mining Software', license='GPLv3', url='', platforms=['x86_64'], ext_modules=[cpp_core], packages=get_packages(), package_dir={'tomominer': 'tomominer', 'tomominer.core': 'tomominer/core/cython/', }, scripts=['bin/tm_workers_local', 'bin/tm_server', 'bin/tm_pose_kmeans', 'bin/tm_pose_norm', 'bin/tm_pose_contigency_table', 'bin/tm_pursuit', 'bin/tm_pursuit_contigency_table', 'bin/tm_refine'], cmdclass={'build_ext': build_ext, })
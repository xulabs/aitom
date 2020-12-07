#!/usr/bin/env python


import os
import platform
# from distutils.core import setup
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from codecs import open
import numpy as N

# Make sure to use gcc compiler instead of x86_64-conda_cos6-linux-gnu-cc or x86_64-linux-gnu-gcc
os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"


compile_extra_args = ['-std=c++11']
link_extra_args = []
if platform.system() == "Darwin":
    compile_extra_args = ['-std=c++11', "-mmacosx-version-min=10.9"]
    link_extra_args = ["-stdlib=libc++", "-mmacosx-version-min=10.9"]


import os
script_dir = os.path.dirname(os.path.realpath(__file__))

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
                     include_dirs=[N.get_include(), '/usr/include',
                                   '/usr/local/include', 'aitom/tomominer/core/src/', os.path.join(script_dir, 'ext', 'include')],      # use script_dir/ext/include to include header files that cannot be installed without root privilege
                     library_dirs=[os.path.join(script_dir, 'ext', 'lib')],     # use script_dir/ext/lib to include library files that cannot be installed without root privilege
                     extra_compile_args=compile_extra_args,
                     extra_link_args=link_extra_args,
                     language='c++')


def get_packages(root_dir='aitom', exclude_dir_roots=['aitom/tomominer/core/src', 'aitom/tomominer/core/cython']):
# def get_packages(root_dir='aitom', exclude_dir_roots=['aitom/tomominer/core']):
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


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(name='aitom',
      version='0.0.1',
      author='Xu Lab (CMU) and collaborators',
      author_email='mxu1@cs.cmu.edu',
      description='AI software for tomogram analysis',
      install_requires=[requirements],
      license='GPLv3',
      url='https://github.com/xulabs/aitom',
      platforms=['x86_64'],
      ext_modules=cythonize(cpp_core),
      packages=get_packages(),
      package_dir={'aitom': 'aitom',
                   'aitom.tomominer.core': 'aitom/tomominer/core/', },
      cmdclass={'build_ext': build_ext, },
      entry_points={
          'console_scripts': [
              'picking = aitom.bin.picking:main',
          ]}
      )

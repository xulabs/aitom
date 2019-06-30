# AITom

AITom is a library for developping AI algorithms for cellular electron cryo-tomography analysis. Developped and maintained by [Xu lab](https://cs.cmu.edu/~mxu1) and collaborators. 

The tomominer module was adapted from an [extended version](http://web.cmb.usc.edu/people/alber/Software/mpp/) of the [tomominer library](https://github.com/alberlab/tomominer).

# Build Tomominer
## Ubuntu

1. install dependencies and build tomominer
```bash
sudo apt-get install python fftw3-dev cython libblas3 liblapack3 python-numpy python-scipy libarmadillo-dev python-sklearn
pip install cython
cd aitom
python setup.py build
```

2. add tomominer in python path.

```bash
export PYTHONPATH=$PYTHONPATH:%aitom_dir%/aitom/build/lib.linux-x86_64-2.7/
```

## Mac OS X

1. Download FFTW3 and install it.

```bash
wget http://www.fftw.org/fftw-3.3.8.tar.gz
tar xvf fftw-3.3.8.tar.gz
cd fftw-3.3.8
./configure
make -j8
sudo make install
```

2. install other dependencies using brew and pip.
```bash
brew install armadillo openblas lapack
pip install cython numpy scipy
```

3. build tomominer

```bash
cd aitom
python setup.py build
```

4. add tomominer in python path.

```bash
export PYTHONPATH=$PYTHONPATH:%aitom_dir%/aitom/build/lib.macosx-10.7-x86_64-2.7/
```



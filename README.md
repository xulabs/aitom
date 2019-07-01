# AITom

AITom is a library for developping AI algorithms for cellular electron cryo-tomography analysis. Developped and maintained by [Xu lab](https://cs.cmu.edu/~mxu1) and collaborators: [Yang lab](http://www.lcecb.org/index.html). 

The tomominer module was adapted from an [extended version](http://web.cmb.usc.edu/people/alber/Software/mpp/) of the [tomominer library](https://github.com/alberlab/tomominer).

# Build library
## Ubuntu

1. install dependencies and build
```bash
sudo apt-get install python fftw3-dev libblas3 liblapack3 libarmadillo-dev
pip install -r requirements.txt
python setup.py build
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
xcode-select --install
brew install armadillo openblas lapack
pip install -r requirements.txt
```

3. build library

```bash
python setup.py build
```


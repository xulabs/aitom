# Build library
## Ubuntu

1. install dependencies, build ext_modules, and install to the system python folder
```bash
sudo apt-get install -y build-essential    # optional step
sudo apt-get install -y python python3-dev fftw3-dev libblas3 liblapack3 libarmadillo-dev
pip install -r requirements.txt
sh clean.sh
sh build.sh
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

3. clean && build library

## Singularity

## Google Colab

Please refer to [Install_for_Google_Colab.ipynb](https://github.com/xulabs/aitom/blob/master/doc/install/Install_for_Google_Colab.ipynb)

## Docker

```bash
cd doc/install/
docker build -t aitom .
```

## Remarks

For using the deep learning functions based on GPU, we recommend to use CUDA version 10 and up, together with cudaNN

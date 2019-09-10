# AITom

## Introduction
AITom is a library for developing AI algorithms for cellular electron cryo-tomography analysis. Developed and maintained by [Xu Lab](https://cs.cmu.edu/~mxu1) and collaborators, particularly [Yang Lab](http://www.lcecb.org/index.html). 

The tomominer module was adapted from an [extended version](http://web.cmb.usc.edu/people/alber/Software/mpp/) of the [tomominer library](https://github.com/alberlab/tomominer), developed at [Alber Lab](http://web.cmb.usc.edu/people/alber/).


# Build library
## Ubuntu

1. install dependencies and build
```bash
sudo apt-get install python python3-dev fftw3-dev libblas3 liblapack3 libarmadillo-dev
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

```bash
sh clean.sh
sh build.sh
```
# About us
## Xulab at Carnegie Mellon University Computational Biology Department
Code and data for projects developed at Xu Lab

The research related to the code and data can be found at http://cs.cmu.edu/~mxu1

### Background
Nearly every major process in a cell is orchestrated by the interplay of macromolecular assemblies, which often coordinate their actions as functional modules in biochemical pathways.  To proceed efficiently,  this  interplay  between  different macromolecular  machines  often  requires  a  distinctly nonrandom spatial organization in the cell. With the recent revolutions in cellular Cryo-Electron Tomography (Cryo-ET) imaging technologies, it is now possible to generate 3D reconstructions of cells in hydrated, close to native states at submolecular resolution. 


### Research
We are developing computational analysis techniques for processing large amounts of Cryo-ET data to reconstruct, detect, classify, recover, and spatially model different cellular components. We utilize state-of-the-art machine learning (including deep learning) approaches to design Cryo-ET specific data analysis and modeling algorithms. Our research automates the cellular structure discovery and will lead to new insights into the basic molecular biology and medical applications.


<img src="https://user-images.githubusercontent.com/31047726/51266413-3613ec00-1989-11e9-810f-f8cb4924f435.png">

*De novo* structural mining pipeline results: (a). A slice of a [rat neuron tomogram](https://doi.org/10.1016/j.cell.2017.12.030),  (b). Recovered patterns (from left to right): mitochondrial membrane, Ribosome-like pattern, ellipsoid of strong signals, TRiC-like pattern, borders of ice crystal, (c). Pattern mining results embedded, (d). Individual patterns embedded.

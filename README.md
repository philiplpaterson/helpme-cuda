<p align="center">
<a href="https://github.com/andysim/helpme/actions"> <img src="https://github.com/andysim/helpme/actions/workflows/build_and_test.yml/badge.svg?branch=master" /></a>
<a href="https://codecov.io/gh/andysim/helpme"> <img src="https://img.shields.io/codecov/c/github/andysim/helpme/master.svg" /></a>
<a href="https://lgtm.com/projects/g/andysim/helpme/context:cpp"><img alt="Language grade: C/C++" src="https://img.shields.io/lgtm/grade/cpp/g/andysim/helpme.svg?logo=lgtm&logoWidth=18"/></a>
<a href="https://lgtm.com/projects/g/andysim/helpme/context:python"><img alt="Language grade: Python" src="https://img.shields.io/lgtm/grade/python/g/andysim/helpme.svg?logo=lgtm&logoWidth=18"/></a>
<a href="https://opensource.org/licenses/BSD-3-Clause"><img src=https://img.shields.io/github/license/andysim/helpme.svg /></a>
</p>

## RPI STUDENT EDITS: ##

Our code is based on an entire repository, so since we cannot link the entire repo, or the files we primarily use (due to size), we have attached our repo link:

https://github.com/philiplpaterson/helpme-cuda


To clone on Aimos, go into scratch and enter these commands:

export http_proxy=http://proxy:8888
export https_proxy=$http_proxy

Then clone the repository using git clone. You can use the makefile inside the TEST folder to compile the code. Compilation warnings will show up, but this is from the repository that we altered. 

If you want to run the CUDA+MPI implementation, use runcuda.sh in either weak.sh or strong.sh (depending on if you want to do weak or strong scaling). If you want to do just MPI implementation, use run.sh.

Then, run weak.sh by saying sh weak.sh. This will release jobs into AIMOS to run your code. Output files will appear matching what you just ran. If you go to the bottom of them, you will see the time each function took and the total time taken to run. 


# About #

**h**elPME: an **e**fficient **l**ibrary for **p**article **m**esh **E**wald.
The recursive acronym is a tip of the hat to early open source software tools
and reflects the recursive algorithms that are key to helPME's support for
arbitrary operators. The library is freely available and is designed to be easy
to use, with minimal setup code required.

## Features ##

* Available as a single C++ header.
* Support for C++/C/Fortran/Python bindings.
* Arbitrary operators including *r*<sup>-1</sup> (Coulomb) and *r*<sup>-6</sup>
  (dispersion).
* Ability to use any floating point precision mode, selectable at run time.
* Three dimensional parallel decomposition with MPI.
* OpenMP parallel threading within each MPI instance (still a work in
  progress).
* Support for arbitrary triclinic lattices and orientations thereof.
* Arbitrary order multipoles supported (still a work in progress).
* Memory for coordinates and forces is taken directly from the caller's pool,
  avoiding copies.

## License ##

helPME is distributed under the
[BSD-3-clause](https://opensource.org/licenses/BSD-3-Clause) open source
license, as described in the LICENSE file in the top level of the repository.
Some external dependencies are used that are licensed under different terms, as
enumerated below.

## Dependencies ##
* Either [FFTW](http://www.fftw.org/)
  [(GPL license)](https://opensource.org/licenses/gpl-license) or
  [MKL](https://software.intel.com/en-us/mkl)
  [(ISSL license)](https://software.intel.com/en-us/license/intel-simplified-software-license)
  required to carry out fast Fourier transforms.
* [CMake](https://cmake.org) required if building the code
  [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [pybind11](https://github.com/pybind/pybind11) required if Python bindings
  are to be built [(BSD-3-clause license)](https://opensource.org/licenses/BSD-3-Clause).
* [Catch2](https://github.com/catchorg/Catch2) for unit testing 
  [(BSL license)](https://opensource.org/licenses/BSL-1.0).

## Requirements ##
helPME is written in C++11, and should work with any modern (well, non-ancient)
C++ compiler.  Python and Fortran bindings are optional, and are built by
default.

## RPI Refactorers ##
Philip Patterson
Sarvesh Sundaram
Ethan Zhang
Daniel He


## Original Authors ##
Andrew C. Simmonett (NIH)
Lori A. Burns (GA Tech)
Daniel R. Roe (NIH)
Bernard R. Brooks (NIH)

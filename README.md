# nnlib - Neural Network Library

nnlib is an all-header library for building, training, and using neural networks.
It is designed to be both easy-to-use and efficient, using BLAS to accelerate math calculations.
nnlib depends on OpenBLAS (on Linux) or the Accelerate framework (on OS X).

# Get Started

Run the following commands to clone the repo and install the headers.

	git clone https://github.com/thelukester92/nnlib.git
	cd nnlib/src
	sudo make install

After nnlib is installed, you can use it right away with `#include <nnlib.h>`.
The default installation directory is `/usr/local/include`. Make sure that is in your 
For a different install directory, use `make install prefix=/path/to/dir`.
When you compile, make sure you link BLAS.
On Linux, you can do this with the `-lopenblas` flag.
On OS X, you can do this with the `-framework Accelerate` flag.
Make sure you use C++11 with the `-std=c++11` flag.

By default, code compiles in debug mode.
To optimize debugging checks out, use `-O3 -DOPTIMIZE`.

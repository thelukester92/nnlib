# Neural Network Library (nnlib)

This is Luke Godfrey's neural network library (nnlib), an all-header library to facilitate the use of artificial neural networks.
It is designed to be fast and easy to use.
nnlib is written entirely in C++ on a Mac, but should work on other platforms as well (official support should come eventually).
You will need to install OpenBLAS and link it on Linux.

# Get Started

Getting started is as easy as cloning the repository and installing. From a terminal:

	git clone https://github.com/thelukester92/nnlib.git
	cd nnlib/src
	sudo make install

After nnlib is installed, you can use it in a new C++ file right away by using `#include <nnlib.h>`.
`make install` will install the header files to `/usr/local/include`.
Make sure that is in your include path (i.e. using a `-I` flag on your compiler).
For a different install directory, use `sudo make install prefix=/path/to/dir`.

When you compile, make sure you link BLAS.
On a Mac, you can do this by adding the `-framework Accelerate` flags to the compile command.
Make sure you use at least C++11 with the `-std=c++11` flag.
By default, code runs in debug mode. To optimize debugging checks out, use the `-O3 -DOPTIMIZE` flags.

# Example

For a complete example, see [the examples repo](https://github.com/thelukester92/nnlib_examples).

# Neural Network Library (nnlib)

This is Luke Godfrey's neural network library (nnlib), an all-header library to facilitate the use of artificial neural networks.
It is designed to be fast and easy to use.
nnlib is written entirely in C++ on a Mac, but should work on other platforms as well (official support should come eventually).

# Get Started

Getting started is as easy as cloning the repository and installing. From a terminal:

	git clone https://github.com/thelukester92/nnlib.git
	cd nnlib/src
	sudo make install

After nnlib is installed, you can use it in a new C++ file right away by using `#include <nnlib.h>`.
`make install` will install the header files to `usr/local/include`.
Make sure that is in your include path (i.e. using a `-I` flag on your compiler).
For a different install directory, use `sudo make install prefix=/path/to/dir`.

# Example

For a complete example, see src/main.cpp.

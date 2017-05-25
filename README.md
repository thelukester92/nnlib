# nnlib - Neural Network Library

[![Build Status](https://travis-ci.org/thelukester92/nnlib.svg?branch=master)](https://travis-ci.org/thelukester92/nnlib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

nnlib is an all-header library for building, training, and using neural networks.
It is designed to be both easy-to-use and efficient, using BLAS to accelerate math calculations.
nnlib has no dependencies, but is compatible with OpenBLAS (Linux) or the Accelerate framework (OS X).
For examples on how to use nnlib, see the [examples repository](https://github.com/thelukester92/nnlib-examples).

This library is *experimental*.
I want it to become a useful, general purpose library to use as an alternative to popular frameworks like Torch, but it is not yet ready.
Use at your own risk, and expect future commits to break your code.

# Get Started

Run the following commands to clone the repo and install the headers.

	git clone https://github.com/thelukester92/nnlib.git
	cd nnlib
	make install

The default installation directory is `/usr/local/include`.
For a different install directory, use `make install PREFIX=/path/to/dir`.
To run unit tests, use `make test`.

To use nnlib in your code, `#include <nnlib.h>` and you're all set!
You must compile with C++11 (`-std=c++11` in most compilers).
To enable acceleration, add the `-DACCELERATE_BLAS` flag and either `-lopenblas` on Linux or `-framework Accelerate` on OS X.

> Note: If you're using gcc (instead of clang) on OS X, you may also need to use the `-flax-vector-conversions` flag.

Runtime checks can be optimized out with the `-DOPTIMIZE` flag.
It is highly recommended that you do *not* use this flag until you are certain your code works.

# Description

## Recurrent Neural networks

Recurrent modules (`Recurrent` and `LSTM`) process a single time step.
Without any extra work, you can feed in a sequence one at a time as to a regular network.
For training, however, backpropagation requires you to keep track of the state (retrieved using the `innerState` method) at each time step.
As an alternative to backing up and restoring state yourself, backpropagation through time has been abstracted away using the `Sequencer` module.
The `Sequencer` module accepts a module (i.e. a recurrent module or a module containing a recurrent module) and a sequence length.
A `Sequencer` is be trained on sequences of batches (`seqLen x batchSize x input/output` tensors).
`Sequencer` takes care of managing state for you, so that an RNN can be trained as follows:

	Sequencer<> rnn(
		new Sequential<>(
			new LSTM<>(1, 10),
			new Linear<>(10, 10),
			new TanH<>()
		),
		seqLen
	);
	MSE<> critic(rnn.outputs());
	SGD<> optimizer(rnn, critic);
	for(size_t epoch = 0; epoch < 100; ++epoch)
		optimizer.step(sequenceIn, sequenceOut);

Without `Sequencer`, training an RNN is much more difficult.

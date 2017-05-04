# nnlib - Neural Network Library

nnlib is an all-header library for building, training, and using neural networks.
It is designed to be both easy-to-use and efficient, using BLAS to accelerate math calculations.
nnlib depends on OpenBLAS (on Linux) or the Accelerate framework (on OS X).
For examples on how to use nnlib, see the [examples repository](https://github.com/thelukester92/nnlib-examples).

# Get Started

Run the following commands to clone the repo and install the headers.

	git clone https://github.com/thelukester92/nnlib.git
	cd nnlib/src
	sudo make install

After nnlib is installed, you can use it right away with `#include <nnlib.h>`.
The default installation directory is `/usr/local/include`.
For a different install directory, use `make install prefix=/path/to/dir`.

When you compile, make sure you link BLAS.
On Linux, you can do this with the `-lopenblas` flag.
On OS X, you can do this with the `-framework Accelerate` flag.

Make sure you use C++11 with the `-std=c++11` flag.

By default, code compiles in debug mode.
To optimize debugging checks out, use `-O3 -DOPTIMIZE`.

# Description

## Recurrent Neural networks

Recurrent modules, such as `Recurrent`, `LSTM`, and `GRU`, process a single time step.
Without any extra work, you can feed in a sequence one at a time as to a regular network.
For training, however, backpropagation requires you to keep track of the state at each time step.
As an alternative to backing up and restoring state using the `innerState` method, backpropagation through time has been abstracted away using the `Sequencer` module.
The `Sequencer` module accepts a module (i.e. a recurrent module) and a sequence length and can be trained on sequences of batches (`seqLen` by `batchSize` by `input/output` tensors).
`Sequencer` takes care of managing state for you, so that an RNN can be trained as follows:

	Sequencer<> rnn(
		new Sequential<>(
			new LSTM<>(1, 10),
			new Linear<>(10, 10),
			new TanH<>()
		),
		seqLen
	);
	MSE<> critic(rnn);
	SGD<Sequencer, MSE> optimizer(rnn, critic);
	for(size_t epoch = 0; epoch < 100; ++epoch)
		optimizer.step(sequenceIn, sequenceOut);

Without `Sequencer`, training an RNN is much more difficult.

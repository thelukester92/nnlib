# nnlib - Neural Network Library

[![Build Status](https://api.travis-ci.org/thelukester92/nnlib.svg?branch=master)](https://travis-ci.org/thelukester92/nnlib)
[![codecov](https://codecov.io/gh/thelukester92/nnlib/branch/master/graph/badge.svg)](https://codecov.io/gh/thelukester92/nnlib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

nnlib is an all-header library for building, training, and using neural networks.
It is designed to be both easy-to-use and efficient, using BLAS to accelerate math calculations.
nnlib has no dependencies, but is compatible with OpenBLAS (Linux) or the Accelerate framework (OS X).

This library is stable, tested, and used in a number of private projects.
Feel free to use it in your own projects, submit issues, or fork it and play with it yourself!

# Get Started

Run the following commands to clone the repo and install the headers.

	git clone https://github.com/thelukester92/nnlib.git
	cd nnlib
	make install

The default installation directory is `/usr/local/include`.
For a different install directory, use `make install PREFIX=/path/to/dir`.
To run unit tests, use `make test`.

To use nnlib in your code, you can include individual files (i.e. `#include <nnlib/nn/linear.hpp>`) or you can include everything by using `#include <nnlib.hpp>`.
You must compile with C++11 (`-std=c++11` in most compilers).
To enable acceleration, add the `-DACCELERATE_BLAS` flag and either `-lopenblas` on Linux or `-framework Accelerate` on OS X.

> Note: If you're using gcc (instead of clang) on OS X, you may also need to use the `-flax-vector-conversions` flag.

Some runtime checks can be optimized out with the `-DOPTIMIZE` flag.
It is highly recommended that you do *not* use this flag until you are certain your code works.

# Examples

Check out the [examples repository](https://github.com/thelukester92/nnlib-examples) for complete examples!

## Simple Neural Network

A standard neural network can be built as a `Sequential` module.
A common topology is a sequence of `Linear` layers with each one followed by a nonlinearity such as `TanH` or `ReLU`.
Here's some sample code for building a neural network for a binary classification problem from 500 inputs to 1 output.

	Sequential<> nn(
		new Linear<>(500, 200), new TanH<>(),
		new Linear<>(200, 100), new TanH<>(),
		new Linear<>(100, 1)
	);

Next, we want to load the data.
This can be done most easily with `CSVSerializer` or `JSONSerializer`.
Let's say we have the data in a file called ``data.csv``, where the first 500 columns are features and the last column is data.
We will also randomly batch the data into batches of size 25 using `Batcher`.

	Serialized rows = CSVSerializer::readFile("data.csv");
	Tensor<> data(rows.size(), rows.get(0)->size());
	for(size_t i = 0; i < rows.size(); ++i)
	{
		Serialized &row = *rows.get(i);
		for(size_t j = 0; j < row.size(); ++j)
		{
			data(i, j) = row.get<double>(j);
		}
	}
	
	Tensor<> features = data.narrow(1, 0, 500), labels = data.narrow(1, 500, 1);
	Batcher<> batcher(features, labels, 25);

Finally, we will create our critic (target function) and optimizer and train for 100 epochs.

	MSE<> critic;
	SGD<> optimizer(nn, critic);
	
	for(size_t i = 0; i < 100; ++i)
	{
		batcher.reset();
		do
		{
			optimizer.step(batcher.features(), batcher.labels());
		}
		while(batcher.next());
	}

Final accuracy can be obtained by using the critic.

	cout << "Final accuracy: " << critic.forward(nn.forward(features), labels) << endl;

## Recurrent Neural networks

Recurrent neural networks process sequences instead of single values.
Most modules, including the recurrent `LSTM` module, process a single time step.
To make it recurrent, wrap each piece of the network in a `Sequencer`, which will handle sequence-based training.

	Sequential<> rnn(
		new Sequencer<>(new LSTM<>(500, 200)),
		new Sequencer<>(new LSTM<>(200, 100)),
		new Sequencer<>(new LSTM<>(100, 1))
	);

Data can be loaded the same way, but must be batched with a `SequenceBatcher`.
Here's what that looks like if our target training sequence length is 50 and batch size is 25.

	SequenceBatcher<> batcher(features, labels, 50, 25);

The critic should be wrapped in `CriticSequencer`.

	CriticSequencer<> critic(new MSE<>());

Finally, training works exactly as it did before, although `SequenceBatcher` works by presentations instead of by epochs.
You may also choose to reset the RNN state before each training presentation.
Here's how to train an RNN for 500 presentations:

	for(size_t i = 0; i < 500; ++i)
	{
		rnn.forget();
		batcher.reset();
		optimizer.step(batcher.features(), batcher.labels());
	}

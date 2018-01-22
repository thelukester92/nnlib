# nnlib - Neural Network Library

[![Build Status](https://api.travis-ci.org/thelukester92/nnlib.svg?branch=master)](https://travis-ci.org/thelukester92/nnlib)
[![codecov](https://codecov.io/gh/thelukester92/nnlib/branch/master/graph/badge.svg)](https://codecov.io/gh/thelukester92/nnlib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

nnlib is an C++ library for building, training, and using neural networks.
It can be used header-only for flexibility or precompiled for rapid development.
While nnlib can be used on its own, we recommend using compatible CPU and GPU libraries for accelerating matrix and vector operations.
nnlib is compatible with the following libraries:

* The [Accelerate framework](https://developer.apple.com/documentation/accelerate) - CPU BLAS on OS X.
* [OpenBLAS](https://github.com/xianyi/openblas) - CPU BLAS on multiple platforms.
* [NVBlas](http://docs.nvidia.com/cuda/nvblas) - GPU BLAS on NVIDIA GPUs.

This library is stable, tested, and used in a number of private projects.
Feel free to use it in your own projects, submit issues, or fork it and play with it yourself!

# Installing on OS X

	brew tap thelukester92/nnlib
	brew install nnlib

# Installing on Linux

	git clone https://github.com/thelukester92/nnlib.git
	cd nnlib
	make && sudo make install

The default installation directory is `/usr/local`.
For a different install directory, use `make install PREFIX=/path/to/dir`.
To install headers only, use `make headers`.
To run unit tests, use `make test`.

# Getting Started

To use nnlib in your code, you can include individual files (i.e. `#include <nnlib/nn/linear.hpp>`) or you can include everything by using `#include <nnlib.hpp>`.
You must compile with C++11 (`-std=c++11` in most compilers).
If you use the shared libraries, link to the installed optimized or debugging library to use nnlib.

When using the header-only version, you *must* compile with `NN_HEADER_ONLY` defined and, optionally, the `NN_ACCEL_CPU` and `NN_ACCEL_GPU` flags to enable linear algebra acceleration.
You may optimize out some runtime asserts with the `-DNN_OPT` flag.
It is highly recommended that you do *not* use this flag until you are certain your code works.

> Note: If you're using gcc (instead of clang) on OS X, you may also need to use the `-flax-vector-conversions` flag.

# Examples

Check out the [examples repository](https://github.com/thelukester92/nnlib-examples) for complete examples!

# Tools

Check out the [tools repository](https://github.com/thelukester92/nnlib-tools) for useful machine learning CLI utilities!

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

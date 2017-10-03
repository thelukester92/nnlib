#ifndef TOY_TIME_SERIES_HPP
#define TOY_TIME_SERIES_HPP

#include "nnlib/core/tensor.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/nn/lstm.hpp"
#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/sequencer.hpp"
#include "nnlib/opt/sgd.hpp"
#include "nnlib/util/batcher.hpp"

void ToyTimeSeries()
{
	const double pi = atan(1) * 4;
	
	Tensor<> train(100, 1), test;
	for(size_t i = 0; i < train.size(0); ++i)
		train(i, 0) = sin(pi * i / 10.0);
	test = train.narrow(0, 50, 50);
	train = train.narrow(0, 0, 50);
	
	Sequencer<> nn(
		new Sequential<>(
			new LSTM<>(1, 10),
			new Linear<>(10, 1)
		)
	);
	
	MSE<> critic;
	SGD<> opt(nn, critic);
	opt.learningRate(1e-3);
	
	SequenceBatcher<> batcher(train.narrow(0, 0, 49), train.narrow(0, 1, 49), 25, 10);
	for(size_t presentation = 0; presentation < 500; ++presentation)
	{
		batcher.reset();
		nn.forget();
		opt.step(batcher.features(), batcher.labels());
	}
	
	// this amount of training should be plenty to fit testing data within 1.0 error
	
	Tensor<> preds(50, 1);
	nn.forget();
	nn.forward(train.view(50, 1, 1));
	for(size_t i = 0; i < preds.size(0); ++i)
	{
		preds(i, 0) = nn.output()(nn.output().size(0) - 1, 0, 0);
		nn.forward(preds.select(0, i).view(1, 1, 1));
	}
	NNAssertLessThan(critic.forward(preds, test), 1.0, "Too much error!");
}

#endif

#ifndef TEST_DROPCONNECT_H
#define TEST_DROPCONNECT_H

#include "nnlib/nn/dropconnect.h"
#include "nnlib/nn/linear.h"
#include "test_module.h"
using namespace nnlib;

/// \brief Test the DropConnect module.
///
/// DropConnect depends on a stochastic process, so testing it is less straightforward.
/// We test by running it several times and making sure the statistics are approximately right.
void TestDropConnect()
{
	RandomEngine::seed(0);
	
	Tensor<> inp = Tensor<>(4, 25).ones();
	Tensor<> grd = Tensor<>(4, 1).ones();
	double p = 0.75, sum1 = 0, sum2 = 0;
	size_t c = 100;
	
	Linear<> *linear = new Linear<>(25, 1);
	linear->weights().ones();
	linear->bias().zeros();
	
	DropConnect<> module(linear, p);
	NNAssertEquals(module.dropProbability(), p, "DropConnect::DropConnect failed!");
	
	for(size_t i = 0; i < c; ++i)
	{
		sum1 += module.forward(inp).sum() / c;
		sum2 += module.backward(inp, grd).sum() / c;
	}
	
	NNAssertAlmostEquals(sum1, c * (1 - p), 2, "DropConnect::forward failed! (Note: this may be due to the random process).");
	NNAssertEquals(sum1, sum2, "DropConnect::backward failed!");
	
	module.dropProbability(p = 0.5).training(false);
	sum1 = sum2 = 0;
	
	for(size_t i = 0; i < c; ++i)
	{
		sum1 += module.forward(inp).sum() / c;
		sum2 += module.backward(inp, grd).sum() / c;
	}
	
	NNAssertAlmostEquals(sum1, c * (1 - p), 2, "DropConnect::forward (inference) failed! (Note: this may be due to the random process).");
	NNAssertEquals(sum1, sum2, "DropConnect::backward (inference) failed!");
	
	module.state().fill(0);
	NNAssertAlmostEquals(module.output().sum(), 0, 1e-12, "DropConnect::state failed!");
	
	{
		auto l1 = module.paramsList();
		auto l2 = linear->paramsList();
		NNAssertEquals(l1.size(), l2.size(), "DropConnect::paramsList failed!");
		for(auto i = l1.begin(), j = l2.begin(), end = l1.end(); i != end; ++i, ++j)
			NNAssertEquals(*i, *j, "DropConnect::paramsList failed!");
	}
	
	{
		auto l1 = module.gradList();
		auto l2 = linear->gradList();
		NNAssertEquals(l1.size(), l2.size(), "DropConnect::gradList failed!");
		for(auto i = l1.begin(), j = l2.begin(), end = l1.end(); i != end; ++i, ++j)
			NNAssertEquals(*i, *j, "DropConnect::gradList failed!");
	}
	
	TestModule("DropConnect", module, inp);
}

#endif

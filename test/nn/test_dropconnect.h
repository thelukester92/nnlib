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
	
	Linear<> *linear = new Linear<>(25, 1, 4);
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
	
	module.batch(32);
	NNAssertEquals(module.batch(), 32, "DropConnect::batch failed!");
	
	module.inputs({ 3, 4 });
	NNAssertEquals(module.inputs(), Storage<size_t>({ 3, 4 }), "DropConnect::inputs failed!");
	NNAssertEquals(module.module().inputs(), module.inputs(), "DropConnect::inputs failed! Wrong inner input shape!");
	
	module.outputs({ 12, 3 });
	NNAssertEquals(module.outputs(), Storage<size_t>({ 12, 3 }), "DropConnect::outputs failed!");
	NNAssertEquals(module.module().outputs(), module.outputs(), "DropConnect::outputs failed! Wrong inner output shape!");
	
	bool ok = true;
	try
	{
		module.add(nullptr);
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "DropConnect::add failed to throw an error!");
	
	ok = true;
	try
	{
		module.remove(0);
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "DropConnect::remove failed to throw an error!");
	
	ok = true;
	try
	{
		module.clear();
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "DropConnect::clear failed to throw an error!");
	
	module.state().fill(0);
	NNAssertAlmostEquals(module.output().sum(), 0, 1e-12, "DropConnect::state failed!");
	
	TestModule(module);
}

#endif

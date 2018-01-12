#include "../test_linear.hpp"
#include "../test_module.hpp"
#include "nnlib/nn/linear.hpp"
using namespace nnlib;

void TestLinear()
{
	// Linear layer with arbitrary parameters
	Linear<NN_REAL_T> module(2, 3);
	module.weights().copy({ -3, -2, 2, 3, 4, 5 });
	module.bias().copy({ -5, 7, 8862.37 });

	// Arbitrary input (batch)
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -5, 10, 15, -20 }).resize(2, 2);

	// Arbitrary output gradient (batch)
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 1, 2, 3, -4, -3, 2 }).resize(2, 3);

	// Output (fixed given input, weights, and bias)
	Tensor<NN_REAL_T> out = Tensor<NN_REAL_T>({ 40, 57, 8902.37, -110, -103, 8792.37 }).resize(2, 3);

	// Input gradient (fixed given input, weights, bias, and output gradient)
	Tensor<NN_REAL_T> ing = Tensor<NN_REAL_T>({ -1, 26, 22, -14 }).resize(2, 2);

	// Parameter gradient (fixed given input and output gradient)
	Tensor<NN_REAL_T> prg = Tensor<NN_REAL_T>({ -65, -55, 15, 90, 80, -10, -3, -1, 5 });

	// Test forward and backward using the parameters and targets above

	module.forward(inp);
	module.backward(inp, grd);

	NNAssertLessThan((module.output() - out).square().sum(), 1e-9, "Linear::forward failed; wrong output!");
	NNAssertLessThan((module.inGrad() - ing).square().sum(), 1e-9, "Linear::backward failed; wrong input gradient!");
	NNAssertLessThan((module.grad() - prg).square().sum(), 1e-9, "Linear::backward failed; wrong parameter gradient!");

	module.forward(inp.select(0, 0));
	module.backward(inp.select(0, 0), grd.select(0, 0));

	NNAssertLessThan(module.output().copy().add(out.select(0, 0), -1).square().sum(), 1e-9, "Linear::forward failed for a vector; wrong output!");
	NNAssertLessThan(module.inGrad().copy().add(ing.select(0, 0), -1).square().sum(), 1e-12, "Linear::backward failed for a vector; wrong input gradient!");

	Linear<NN_REAL_T> unbiased(2, 3, false);
	unbiased.weights().copy(module.weights());

	unbiased.forward(inp.select(0, 0));
	unbiased.backward(inp.select(0, 0), grd.select(0, 0));

	NNAssertLessThan(unbiased.output().copy().add(module.bias()).add(out.select(0, 0), -1).square().sum(), 1e-9, "Linear::forward failed without bias; wrong output!");
	NNAssertLessThan(unbiased.inGrad().copy().add(ing.select(0, 0), -1).square().sum(), 1e-9, "Linear::backward failed without bias; wrong input gradient!");

	bool ok = true;
	try
	{
		module.forward(Tensor<NN_REAL_T>(1, 1, 1));
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "Linear::forward accepted an invalid input shape!");

	ok = true;
	try
	{
		module.backward(Tensor<NN_REAL_T>(1, 1, 1), Tensor<NN_REAL_T>(1, 1, 1));
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "Linear::backward accepted invalid input and outGrad shapes!");

	TestModule("Linear", module, inp);
}

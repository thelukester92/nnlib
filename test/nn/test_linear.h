#ifndef TEST_LINEAR_H
#define TEST_LINEAR_H

#include "nnlib/nn/linear.h"
#include "test_module.h"
using namespace nnlib;

void TestLinear()
{
	// Linear layer with arbitrary parameters
	Linear<> module(2, 3);
	module.weights().copy({ -3, -2, 2, 3, 4, 5 });
	module.bias().copy({ -5, 7, 8862.37 });
	
	// Arbitrary input (batch)
	Tensor<> inp = Tensor<>({ -5, 10, 15, -20 }).resize(2, 2);
	
	// Arbitrary output gradient (batch)
	Tensor<> grd = Tensor<>({ 1, 2, 3, -4, -3, 2 }).resize(2, 3);
	
	// Output (fixed given input, weights, and bias)
	Tensor<> out = Tensor<>({ 40, 57, 8902.37, -110, -103, 8792.37 }).resize(2, 3);
	
	// Input gradient (fixed given input, weights, bias, and output gradient)
	Tensor<> ing = Tensor<>({ -1, 26, 22, -14 }).resize(2, 2);
	
	// Parameter gradient (fixed given input and output gradient)
	Tensor<> prg = Tensor<>({ -65, -55, 15, 90, 80, -10, -3, -1, 5 });
	
	// Test forward and backward using the parameters and targets above
	
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssert(module.output().copy().addM(out, -1).square().sum() < 1e-9, "Linear::forward failed; wrong output!");
	NNAssert(module.inGrad().copy().addM(ing, -1).square().sum() < 1e-9, "Linear::backward failed; wrong input gradient!");
	NNAssert(module.grad().addV(prg, -1).square().sum() < 1e-9, "Linear::backward failed; wrong parameter gradient!");
	
	TestModule("Linear", module, inp);
}

#endif

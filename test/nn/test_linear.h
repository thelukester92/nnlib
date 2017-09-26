#ifndef TEST_LINEAR_H
#define TEST_LINEAR_H

#include "nnlib/nn/linear.h"
// #include "test_module.h"
using namespace nnlib;

void TestLinear()
{
	Linear<> linear;
	
	/*
	// Input, arbitrary
	Tensor<> inp = Tensor<>({
		-5, 10,
		15, -20
	}).resize(2, 2);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({
		1, 2, 3,
		-4, -3, 2,
	}).resize(2, 3);
	
	// Linear layer with weights and bias, arbitrary
	Linear<> module(2, 3);
	const_cast<Tensor<> &>(module.weights()).copy({
		-3, -2, 2,
		3, 4, 5
	});
	const_cast<Tensor<> &>(module.bias()).copy({ -5, 7, 8862.37 });
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({
		40, 57, 8902.37,
		-110, -103, 8792.37,
	}).resize(2, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({
		-1, 26,
		22, -14
	}).resize(2, 2);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		-65, -55, 15,
		90, 80, -10,
		-3, -1, 5
	});
	
	// Test forward and backward using the parameters and targets above
	
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssert(module.output().copy().addM(out, -1).square().sum() < 1e-9, "Linear::forward failed!");
	NNAssert(module.inGrad().copy().addM(ing, -1).square().sum() < 1e-9, "Linear::backward failed; wrong inGrad!");
	NNAssert(module.grad().addV(prg, -1).square().sum() < 1e-9, "Linear::backward failed; wrong grad!");
	
	Storage<size_t> dims = { 5 };
	
	module.resizeInputs(dims);
	NNAssertEquals(module.inputShape(), dims, "Linear::inputs failed!");
	
	module.resizeOutputs(dims);
	NNAssertEquals(module.outputShape(), dims, "Linear::outputs failed!");
	
	module.resize({ 1 }, { 3 });
	NNAssertEquals(module.inputShape(), Storage<size_t>({ 1 }), "Linear::resize failed; wrong inputs!")
	NNAssertEquals(module.outputShape(), Storage<size_t>({ 3 }), "Linear::resize failed; wrong outputs!")
	
	TestModule(module);
	*/
}

#endif

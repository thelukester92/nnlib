#ifndef TEST_SEQUENTIAL_H
#define TEST_SEQUENTIAL_H

#include "nnlib/nn/sequential.h"
#include "nnlib/nn/linear.h"
#include "nnlib/nn/tanh.h"
using namespace nnlib;

void TestSequential()
{
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
	Linear<> *linear = new Linear<>(2, 3);
	linear->weights().copy({
		-3, -2, 2,
		3, 4, 5
	});
	linear->bias().copy({ -5, 7, 8862.37 });
	
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
	
	Sequential<> module(linear);
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssert(module.output().addM(out, -1).square().sum() < 1e-9, "Sequential::forward failed!");
	NNAssert(module.inGrad().addM(ing, -1).square().sum() < 1e-9, "Sequential::backward failed; wrong inGrad!");
	NNAssert(module.grad().addV(prg, -1).square().sum() < 1e-9, "Sequential::backward failed; wrong grad!");
	
	NNAssert(module.component(0) == linear, "Sequential::component failed to get the correct component!");
	NNAssert(module.components() == 1, "Sequential::components failed!");
	NNAssert(module.remove(0) == linear, "Sequential::remove failed to return the removed component!");
	
	module.clear();
	NNAssert(module.components() == 0, "Sequential::clear failed!");
	
	module.add(linear);
	NNAssert(module.paramsList() == linear->paramsList(), "Sequential::paramsList failed!");
	NNAssert(module.gradList() == linear->gradList(), "Sequential::gradList failed!");
	NNAssert(module.stateList() == linear->stateList(), "Sequential::stateList failed!");
}

#endif

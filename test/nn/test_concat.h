#ifndef TEST_CONCAT_H
#define TEST_CONCAT_H

#include "nnlib/nn/concat.h"
#include "nnlib/nn/linear.h"
#include "nnlib/nn/identity.h"
using namespace nnlib;

void TestConcat()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({
		-5, 10,
		15, -20
	}).resize(2, 2);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({
		1, 2, 3, 4, -5,
		-4, -3, 2, 1, 7
	}).resize(2, 5);
	
	// Linear layer with weights and bias, arbitrary
	Linear<> *linear = new Linear<>(2, 3, 2);
	linear->weights().copy({
		-3, -2, 2,
		3, 4, 5
	});
	linear->bias().copy({ -5, 7, 8862.37 });
	
	// Identity layer
	Identity<> *identity = new Identity<>(2, 2);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({
		40, 57, 8902.37, -5, 10,
		-110, -103, 8792.37, 15, -20
	}).resize(2, 5);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({
		3, 21,
		23, -7
	}).resize(2, 2);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		-65, -55, 15,
		90, 80, -10,
		-3, -1, 5
	});
	
	// Test forward and backward using the parameters and targets above
	
	Concat<> module(linear, identity);
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssert(module.output().addM(out, -1).square().sum() < 1e-9, "Concat::forward failed!");
	NNAssert(module.inGrad().addM(ing, -1).square().sum() < 1e-9, "Concat::backward failed; wrong inGrad!");
	NNAssert(module.grad().addV(prg, -1).square().sum() < 1e-9, "Concat::backward failed; wrong grad!");
	
	NNAssert(module.component(0) == linear, "Concat::component failed to get the correct component!");
	NNAssert(module.component(1) == identity, "Concat::component failed to get the correct component!");
	
	NNAssert(module.components() == 2, "Concat::components failed!");
	
	NNAssert(module.remove(0) == linear, "Concat::remove failed to return the removed component!");
	NNAssert(module.components() == 1, "Concat::remove failed!");
	NNAssert(module.component(0) == identity, "Concat::remove failed!");
	NNAssert(module.outputs() == identity->outputs(), "Concat::remove failed!");
	
	module.clear();
	NNAssert(module.components() == 0, "Concat::clear failed!");
	
	module.add(linear);
	NNAssert(module.outputs() == linear->outputs(), "Concat::add failed!");
	
	module.batch(32);
	NNAssert(module.batch() == 32, "Concat::batch failed to batch container!");
	NNAssert(linear->batch() == 32, "Concat::batch failed to batch children!");
	
	NNAssert(module.parameterList() == linear->parameterList(), "Concat::parameterList failed!");
	NNAssert(module.gradList() == linear->gradList(), "Concat::gradList failed!");
	NNAssert(module.stateList() == linear->stateList(), "Concat::stateList failed!");
	
	TestSerializationOfModule(module);
}

#endif

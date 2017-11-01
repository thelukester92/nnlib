#ifndef TEST_SEQUENTIAL_H
#define TEST_SEQUENTIAL_H

#include "nnlib/nn/sequential.hpp"
#include "nnlib/nn/batchnorm.hpp"
#include "nnlib/nn/linear.hpp"
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
		1, 2,
		-4, -3,
	}).resize(2, 2);
	
	// Linear layer with weights and bias, arbitrary
	Linear<> *linear = new Linear<>(2, 3);
	linear->weights().copy({
		-0.50667886785403, 0.32759806987449, 0.65755833165511,
		0.10750948608753, -0.4340286671044, 0.23516327870398
	});
	linear->bias().copy({ -0.025146033726567, 0.6293391970186, -0.60999610504332 });
	
	// Second linear layer with weights and bias, arbitrary
	Linear<> *linear2 = new Linear<>(3, 2);
	linear2->weights().copy({
		-0.26131507397613, -0.12238678004357,
		-0.25173198611324, -0.52631174306551,
		-0.15799479364335, 0.076954514512593
	});
	linear2->bias().copy({ 0.089687510972433, -0.1780101224473 });
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({
		0.74408910465583, 2.0796612294443,
		-1.6553227544948, -6.1176610608337
	}).resize(2, 2);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({
		-0.17356654756232, 0.510757516282,
		0.39523702099164, -0.87616246292012
	}).resize(2, 2);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		23.716752710845, 45.309724965964, 6.037163288625,
		-33.309299061337, -64.760818195432, -8.0631702668939,
		0.90633200197196, 1.2815077014052, 0.39702986641745,
		42.68541825957, 36.493242652701,
		-62.244472172293, -53.369526408467,
		-19.746608159094, -16.742649839669,
		-3, -1
	});
	
	// Test forward and backward using the parameters and targets above
	
	Sequential<> module(linear, linear2);
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssert(module.output().addM(out, -1).square().sum() < 1e-9, "Sequential::forward failed!");
	NNAssert(module.inGrad().addM(ing, -1).square().sum() < 1e-9, "Sequential::backward failed; wrong inGrad!");
	NNAssert(module.grad().addV(prg, -1).square().sum() < 1e-9, "Sequential::backward failed; wrong grad!");
	
	NNAssert(module.component(0) == linear, "Sequential::component failed to get the correct component!");
	NNAssert(module.components() == 2, "Sequential::components failed!");
	NNAssert(module.remove(0) == linear, "Sequential::remove failed to return the removed component!");
	
	module.clear();
	NNAssert(module.components() == 0, "Sequential::clear failed!");
	
	module.add(linear);
	NNAssert(module.paramsList() == linear->paramsList(), "Sequential::paramsList failed!");
	NNAssert(module.gradList() == linear->gradList(), "Sequential::gradList failed!");
	
	{
		BatchNorm<> *b = new BatchNorm<>(10);
		Sequential<> s(b);
		s.training(false);
		NNAssert(!b->isTraining(), "Sequential::training failed!");
	}
	
	Sequential<> module2(new Linear<>(5, 10), new Linear<>(10, 2));
	TestContainer("Sequential", module2, Tensor<>(100, 5).rand());
}

#endif

#ifndef TEST_CONCAT_H
#define TEST_CONCAT_H

#include "nnlib/nn/concat.hpp"
#include "nnlib/nn/linear.hpp"
#include "test_container.hpp"
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
		1, 2, 3, 4, 5, -6,
		-4, -3, 2, 1, -1, 2
	}).resize(2, 6);
	
	// Linear layer with weights and bias, arbitrary
	Linear<> *linear = new Linear<>(2, 3);
	linear->weights().copy({
		-0.50667886785403, 0.32759806987449, 0.65755833165511,
		0.10750948608753, -0.4340286671044, 0.23516327870398
	});
	linear->bias().copy({ -0.025146033726567, 0.6293391970186, -0.60999610504332 });
	
	// Second linear layer with weights and bias, arbitrary
	Linear<> *linear2 = new Linear<>(2, 3);
	linear2->weights().copy({
		-0.26131507397613, -0.25173198611324, -0.15799479364335,
		-0.12238678004357, -0.52631174306551, 0.076954514512593
	});
	linear2->bias().copy({ 0.089687510972433, -0.1780101224473, 0.5643456848455 });
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({
		3.5833431664189, -5.3489378233979, -1.5461549762791, 0.17239508041738, -4.1824676225362, 2.1238647981882,
		-9.7755187732876, 14.223883587224, 4.5501132957037, -1.3823029977981, 6.5722449471643, -3.3446665100566
	}).resize(2, 6);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({
		0.76524080224966, -3.6378909345867,
		2.0334652499533, 1.9002086064182
	}).resize(2, 2);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		-65, -55, 15,
		90, 80, -10,
		-3, -1, 5,
		-5, -40, 60,
		20, 70, -100,
		5, 4, -4
	});
	
	// Test forward and backward using the parameters and targets above
	
	Concat<> module(linear, linear2);
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssert(module.output().addM(out, -1).square().sum() < 1e-9, "Concat::forward failed!");
	NNAssert(module.inGrad().addM(ing, -1).square().sum() < 1e-9, "Concat::backward failed; wrong inGrad!");
	NNAssert(module.grad().addV(prg, -1).square().sum() < 1e-9, "Concat::backward failed; wrong grad!");
	
	NNAssert(module.component(0) == linear, "Concat::component failed to get the correct component!");
	NNAssert(module.components() == 2, "Concat::components failed!");
	NNAssert(module.remove(0) == linear, "Concat::remove failed to return the removed component!");
	
	module.clear();
	NNAssert(module.components() == 0, "Concat::clear failed!");
	
	module.add(linear);
	NNAssert(module.paramsList() == linear->paramsList(), "Concat::paramsList failed!");
	NNAssert(module.gradList() == linear->gradList(), "Concat::gradList failed!");
	NNAssert(module.stateList() == linear->stateList(), "Concat::stateList failed!");
	
	TestContainer("Concat", module, inp);
}

#endif

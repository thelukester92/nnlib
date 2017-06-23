#ifndef TEST_RELU_H
#define TEST_RELU_H

#include "nnlib/nn/relu.h"
#include "test_module.h"
using namespace nnlib;

void TestReLU()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = inp.copy();
	out(0, 0) = 0;
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = grd.copy();
	ing(0, 0) = 0;
	
	// Begin test
	
	ReLU<> map(3, 1);
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().addM(out, -1).square().sum() < 1e-9, "ReLU::forward failed!");
	NNAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "ReLU::backward failed!");
	
	map.leak(0.1479);
	out(0, 0) = inp(0, 0) * 0.1479;
	ing(0, 0) = grd(0, 0) * 0.1479;
	
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().addM(out, -1).square().sum() < 1e-9, "ReLU::forward (leaky) failed!");
	NNAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "ReLU::backward (leaky) failed!");
	
	map.inputs({ 3, 4 });
	NNAssert(map.inputs() == map.outputs(), "ReLU::inputs failed to resize outputs!");
	
	map.outputs({ 12, 3 });
	NNAssert(map.inputs() == map.outputs(), "ReLU::outputs failed to resize inputs!");
	
	bool ok = true;
	try
	{
		map.resize({ 3, 4 }, { 4, 3 });
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "ReLU::resize allowed unequal inputs and outputs!");
	
	TestSerializationOfModule(map);
	TestModule(map);
}

#endif

#ifndef TEST_TANH_H
#define TEST_TANH_H

#include "nnlib/nn/tanh.hpp"
#include "test_map.hpp"
using namespace nnlib;

void TestTanH()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({ -0.86172315931, 0.76159415595, 0.99626020494 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ 0.5148663934, -1.25992302484, 0.00746560404 }).resize(1, 3);
	
	TanH<> map;
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().copy().addM(out, -1).square().sum() < 1e-9, "TanH::forward failed!");
	NNAssert(map.inGrad().copy().addM(ing, -1).square().sum() < 1e-9, "TanH::backward failed!");
	
	TestMap("TanH", map, inp);
}

#endif

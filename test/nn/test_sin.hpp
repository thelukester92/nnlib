#ifndef TEST_SIN_H
#define TEST_SIN_H

#include "nnlib/nn/sin.hpp"
#include "test_map.hpp"
using namespace nnlib;

void TestSin()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({ -0.96355818541, 0.8414709848, 0.00159265291 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ 0.53499765724, -1.6209069176, -0.99999873172 }).resize(1, 3);
	
	Sin<> map;
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().copy().addM(out, -1).square().sum() < 1e-9, "Sin::forward failed!");
	NNAssert(map.inGrad().copy().addM(ing, -1).square().sum() < 1e-9, "Sin::backward failed!");
	
	TestMap("Sin", map, inp);
}

#endif

#ifndef TEST_RELU_H
#define TEST_RELU_H

#include "nnlib/nn/relu.h"
#include "test_map.h"
using namespace nnlib;

void TestReLU()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = inp.copy();
	out(0, 0) *= 0.5;
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = grd.copy();
	ing(0, 0) *= 0.5;
	
	ReLU<> map(0.5);
	TestMap("ReLU", map, inp, grd, out, ing);
}

#endif

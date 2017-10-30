#ifndef TEST_SOFTMAX_H
#define TEST_SOFTMAX_H

#include "nnlib/nn/softmax.hpp"
#include "test_module.hpp"
using namespace nnlib;

void TestSoftMax()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -4, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({ 0.01044395976, 0.10416996025, 0.88538607998 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ 0.01577461784, -0.46768084505, 0.45190622721 }).resize(1, 3);
	
	// Begin test
	
	SoftMax<> map;
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().addM(out, -1).square().sum() < 1e-9, "SoftMax::forward failed!");
	NNAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "SoftMax::backward failed!");
	
	TestModule("SoftMax", map, inp);
}

#endif

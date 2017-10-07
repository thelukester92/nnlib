#ifndef TEST_IDENTITY_H
#define TEST_IDENTITY_H

#include "nnlib/nn/identity.hpp"
#include "test_module.hpp"

void TestIdentity()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	Identity<> map;
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().copy().addM(inp, -1).square().sum() < 1e-9, "Identity::forward failed!");
	NNAssert(map.inGrad().copy().addM(grd, -1).square().sum() < 1e-9, "Identity::backward failed!");
	
	TestModule("Identity", map, inp);
}

#endif

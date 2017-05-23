#include "nnlib/nn/tanh.h"
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
	
	// Begin test
	
	TanH<> map(3, 1);
	map.forward(inp);
	map.backward(inp, grd);
	
	NNHardAssert(map.output().addM(out, -1).square().sum() < 1e-9, "TanH::forward failed!");
	NNHardAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "TanH::backward failed!");
}

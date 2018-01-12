#include "../test_map.hpp"
#include "../test_tanh.hpp"
#include "nnlib/nn/tanh.hpp"
using namespace nnlib;

void TestTanH()
{
	// Input, arbitrary
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);

	// Output gradient, arbitrary
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -3, 1 }).resize(1, 3);

	// Output, fixed given input
	Tensor<NN_REAL_T> out = Tensor<NN_REAL_T>({ -0.86172315931, 0.76159415595, 0.99626020494 }).resize(1, 3);

	// Input gradient, fixed given input and output gradient
	Tensor<NN_REAL_T> ing = Tensor<NN_REAL_T>({ 0.5148663934, -1.25992302484, 0.00746560404 }).resize(1, 3);

	TanH<NN_REAL_T> map;
	map.forward(inp);
	map.backward(inp, grd);

	NNAssert((map.output() - out).square().sum() < 1e-9, "TanH::forward failed!");
	NNAssert((map.inGrad() - ing).square().sum() < 1e-9, "TanH::backward failed!");

	TestMap("TanH", map, inp);
}

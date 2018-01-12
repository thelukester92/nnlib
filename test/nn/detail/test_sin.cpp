#include "../test_map.hpp"
#include "../test_sin.hpp"
#include "nnlib/nn/sin.hpp"
using namespace nnlib;

void TestSin()
{
	// Input, arbitrary
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);

	// Output gradient, arbitrary
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -3, 1 }).resize(1, 3);

	// Output, fixed given input
	Tensor<NN_REAL_T> out = Tensor<NN_REAL_T>({ -0.96355818541, 0.8414709848, 0.00159265291 }).resize(1, 3);

	// Input gradient, fixed given input and output gradient
	Tensor<NN_REAL_T> ing = Tensor<NN_REAL_T>({ 0.53499765724, -1.6209069176, -0.99999873172 }).resize(1, 3);

	Sin<NN_REAL_T> map;
	map.forward(inp);
	map.backward(inp, grd);

	NNAssert((map.output() - out).square().sum() < 1e-9, "Sin::forward failed!");
	NNAssert((map.inGrad() - ing).square().sum() < 1e-9, "Sin::backward failed!");

	TestMap("Sin", map, inp);
}

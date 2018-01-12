#include "../test_map.hpp"
#include "../test_elu.hpp"
#include "nnlib/nn/elu.hpp"
#include <math.h>
using namespace nnlib;

void TestELU()
{
	// Input, arbitrary
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);

	// Output gradient, arbitrary
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -3, 1 }).resize(1, 3);

	// Output, fixed given input
	Tensor<NN_REAL_T> out = inp.copy();
	out(0, 0) = 0.5 * (exp(out(0, 0)) - 1);

	// Input gradient, fixed given input and output gradient
	Tensor<NN_REAL_T> ing = grd.copy();
	ing(0, 0) *= out(0, 0) + 0.5;

	ELU<NN_REAL_T> map(0.5);
	map.forward(inp);
	map.backward(inp, grd);

	NNAssert((map.output() - out).square().sum() < 1e-9, "ELU::forward failed!");
	NNAssert((map.inGrad() - ing).square().sum() < 1e-9, "ELU::backward failed!");

	TestMap("ELU", map, inp);
}

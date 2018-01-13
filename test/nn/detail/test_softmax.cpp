#include "../test_module.hpp"
#include "../test_softmax.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/softmax.hpp"
using namespace nnlib;

void TestSoftMax()
{
	// Input, arbitrary
	Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);

	// Output gradient, arbitrary
	Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -4, 1 }).resize(1, 3);

	// Output, fixed given input
	Tensor<NN_REAL_T> out = Tensor<NN_REAL_T>({ 0.01044395976, 0.10416996025, 0.88538607998 }).resize(1, 3);

	// Input gradient, fixed given input and output gradient
	Tensor<NN_REAL_T> ing = Tensor<NN_REAL_T>({ 0.01577461784, -0.46768084505, 0.45190622721 }).resize(1, 3);

	// Begin test

	SoftMax<NN_REAL_T> map;
	map.forward(inp);
	map.backward(inp, grd);

	NNAssert(math::sum(math::square(map.output() - out)) < 1e-9, "SoftMax::forward failed!");
	NNAssert(math::sum(math::square(map.inGrad() - ing)) < 1e-9, "SoftMax::backward failed!");

	TestModule("SoftMax", map, inp);
}
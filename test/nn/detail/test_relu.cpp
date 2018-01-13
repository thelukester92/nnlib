#include "../test_map.hpp"
#include "../test_relu.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/relu.hpp"
using namespace nnlib;

void TestReLU()
{
    // Input, arbitrary
    Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);

    // Output gradient, arbitrary
    Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -3, 1 }).resize(1, 3);

    // Output, fixed given input
    Tensor<NN_REAL_T> out = inp.copy();
    out(0, 0) *= 0.5;

    // Input gradient, fixed given input and output gradient
    Tensor<NN_REAL_T> ing = grd.copy();
    ing(0, 0) *= 0.5;

    ReLU<NN_REAL_T> map(0.5);
    map.forward(inp);
    map.backward(inp, grd);

    NNAssert(math::sum(math::square(map.output() - out)) < 1e-9, "ReLU::forward failed!");
    NNAssert(math::sum(math::square(map.inGrad() - ing)) < 1e-9, "ReLU::backward failed!");

    TestMap("ReLU", map, inp);
}

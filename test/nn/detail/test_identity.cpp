#include "../test_identity.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/identity.hpp"
using namespace nnlib;

void TestIdentity()
{
    // Input, arbitrary
    Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({ -1.3, 1.0, 3.14 }).resize(1, 3);

    // Output gradient, arbitrary
    Tensor<NN_REAL_T> grd = Tensor<NN_REAL_T>({ 2, -3, 1 }).resize(1, 3);

    Identity<NN_REAL_T> map;
    map.forward(inp);
    map.backward(inp, grd);

    NNAssert(math::sum(math::square(map.output() - inp)) < 1e-9, "Identity::forward failed!");
    NNAssert(math::sum(math::square(map.inGrad() - grd)) < 1e-9, "Identity::backward failed!");

    TestModule("Identity", map, inp);
}

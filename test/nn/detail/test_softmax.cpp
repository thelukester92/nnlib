#include "../test_module.hpp"
#include "../test_softmax.hpp"
#include "nnlib/nn/softmax.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(SoftMax)
{
    NNRunAbstractTest(Module, SoftMax, new SoftMax<T>());

    NNTestMethod(forward)
    {
        SoftMax<T> module;
        auto input = Tensor<T>({ -1.3, 1.0, 3.14 }).resize(1, 3);
        module.forward(input);
        NNTestAlmostEquals(module.output()(0, 0), 0.01044395976, 1e-9);
        NNTestAlmostEquals(module.output()(0, 1), 0.10416996025, 1e-9);
        NNTestAlmostEquals(module.output()(0, 2), 0.88538607998, 1e-9);
    }

    NNTestMethod(backward)
    {
        SoftMax<T> module;
        auto input = Tensor<T>({ -1.3, 1.0, 3.14 }).resize(1, 3);
        auto blame = Tensor<T>({ 2, -4, 1 }).resize(1, 3);
        module.forward(input);
        module.backward(input, blame);
        NNTestAlmostEquals(module.inGrad()(0, 0), 0.01577461784, 1e-9);
        NNTestAlmostEquals(module.inGrad()(0, 1), -0.46768084505, 1e-9);
        NNTestAlmostEquals(module.inGrad()(0, 2), 0.45190622721, 1e-9);
    }
}

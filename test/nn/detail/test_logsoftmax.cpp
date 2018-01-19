#include "../test_logsoftmax.hpp"
#include "../test_module.hpp"
#include "nnlib/nn/logsoftmax.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(LogSoftMax)
{
    NNRunAbstractTest(Module, LogSoftMax, new LogSoftMax<T>());

    NNTestMethod(forward)
    {
        LogSoftMax<T> module;
        auto input = Tensor<T>({ -1.3, 1.0, 3.14 }).resize(1, 3);
        module.forward(input);
        NNTestAlmostEquals(module.output()(0, 0), -4.56173148054, 1e-9);
        NNTestAlmostEquals(module.output()(0, 1), -2.26173148054, 1e-9);
        NNTestAlmostEquals(module.output()(0, 2), -0.12173148053, 1e-9);
    }

    NNTestMethod(backward)
    {
        LogSoftMax<T> module;
        auto input = Tensor<T>({ -1.3, 1.0, 3.14 }).resize(1, 3);
        auto blame = Tensor<T>({ 2, -4, 1 }).resize(1, 3);
        module.forward(input);
        module.backward(input, blame);
        NNTestAlmostEquals(module.inGrad()(0, 0), 2.01044395977, 1e-9);
        NNTestAlmostEquals(module.inGrad()(0, 1), -3.89583003975, 1e-9);
        NNTestAlmostEquals(module.inGrad()(0, 2), 1.88538607998, 1e-9);
    }
}

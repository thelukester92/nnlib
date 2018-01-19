#include "../test_logistic.hpp"
#include "../test_map.hpp"
#include "nnlib/nn/logistic.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Logistic)
{
    NNRunAbstractTest(Map, Logistic, new Logistic<T>());

    NNTestMethod(forward)
    {
        Logistic<T> module;
        module.forward({ -1.3, 1.0, 3.14 });
        NNTestAlmostEquals(module.output()(0), 0.21416501695, 1e-9);
        NNTestAlmostEquals(module.output()(1), 0.73105857863, 1e-9);
        NNTestAlmostEquals(module.output()(2), 0.95851288069, 1e-9);
    }

    NNTestMethod(backward)
    {
        Logistic<T> module;
        module.forward({ -1.3, 1.0, 3.14 });
        module.backward({ -1.3, 1.0, 3.14 }, { 2, -3, 1 });
        NNTestAlmostEquals(module.inGrad()(0), 0.33659672493, 1e-9);
        NNTestAlmostEquals(module.inGrad()(1), -0.58983579972, 1e-9);
        NNTestAlmostEquals(module.inGrad()(2), 0.03976593824, 1e-9);
    }
}

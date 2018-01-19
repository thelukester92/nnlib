#include "../test_map.hpp"
#include "../test_relu.hpp"
#include "nnlib/nn/relu.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(ReLU)
{
    NNRunAbstractTest(Map, ReLU, new ReLU<T>());

    NNTestMethod(operator=)
    {
        NNTestParams(const ReLU &)
        {
            ReLU<T> orig(0.7);
            ReLU<T> copy(0.3);
            copy = orig;
            NNTestAlmostEquals(orig.leak(), copy.leak(), 1e-12);
        }
    }

    NNTestMethod(forward)
    {
        ReLU<T> module(0.75);
        module.forward({ -1.3, 1.0, 3.14 });
        NNTestAlmostEquals(module.output()(0), -0.975, 1e-12);
        NNTestAlmostEquals(module.output()(1), 1.0, 1e-12);
        NNTestAlmostEquals(module.output()(2), 3.14, 1e-12);
    }

    NNTestMethod(backward)
    {
        ReLU<T> module(0.75);
        module.forward({ -1.3, 1.0, 3.14 });
        module.backward({ -1.3, 1.0, 3.14 }, { 2, -3, 1 });
        NNTestAlmostEquals(module.inGrad()(0), 1.5, 1e-12);
        NNTestAlmostEquals(module.inGrad()(1), -3, 1e-12);
        NNTestAlmostEquals(module.inGrad()(2), 1, 1e-12);
    }
}

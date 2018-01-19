#include "../test_map.hpp"
#include "../test_elu.hpp"
#include "nnlib/nn/elu.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(ELU)
{
    NNRunAbstractTest(Map, ELU, new ELU<T>());

    NNTestMethod(operator=)
    {
        NNTestParams(const ELU &)
        {
            ELU<T> orig(0.7);
            ELU<T> copy(0.3);
            copy = orig;
            NNTestAlmostEquals(orig.alpha(), copy.alpha(), 1e-12);
        }
    }

    NNTestMethod(forward)
    {
        ELU<T> module(0.5);
        module.forward({ -1.3, 1.0, 3.14 });
        NNTestAlmostEquals(module.output()(0), -0.36373410348, 1e-9);
        NNTestAlmostEquals(module.output()(1), 1.0, 1e-12);
        NNTestAlmostEquals(module.output()(2), 3.14, 1e-12);
    }

    NNTestMethod(backward)
    {
        ELU<T> module(0.5);
        module.forward({ -1.3, 1.0, 3.14 });
        module.backward({ -1.3, 1.0, 3.14 }, { 2, -3, 1 });
        NNTestAlmostEquals(module.inGrad()(0), 0.27253179304, 1e-9);
        NNTestAlmostEquals(module.inGrad()(1), -3, 1e-12);
        NNTestAlmostEquals(module.inGrad()(2), 1, 1e-12);
    }
}

#include "../test_map.hpp"
#include "../test_tanh.hpp"
#include "nnlib/nn/tanh.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(TanH)
{
    NNRunAbstractTest(Map, TanH, new TanH<T>());

    NNTestMethod(forward)
    {
        TanH<T> module;
        module.forward({ -1.3, 1.0, 3.14 });
        NNTestAlmostEquals(module.output()(0), -0.86172315931, 1e-9);
        NNTestAlmostEquals(module.output()(1), 0.76159415595, 1e-9);
        NNTestAlmostEquals(module.output()(2), 0.99626020494, 1e-9);
    }

    NNTestMethod(backward)
    {
        TanH<T> module;
        module.forward({ -1.3, 1.0, 3.14 });
        module.backward({ -1.3, 1.0, 3.14 }, { 2, -3, 1 });
        NNTestAlmostEquals(module.inGrad()(0), 0.5148663934, 1e-9);
        NNTestAlmostEquals(module.inGrad()(1), -1.25992302484, 1e-9);
        NNTestAlmostEquals(module.inGrad()(2), 0.00746560404, 1e-9);
    }
}

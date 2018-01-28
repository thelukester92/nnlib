#include "../test_map.hpp"
#include "../test_sin.hpp"
#include "nnlib/nn/sin.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Sin)
{
    NNRunAbstractTest(Map, Sin, new Sin<T>());

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            Sin<T> module;
            module.forward({ -1.3, 1.0, 3.14 });
            NNTestAlmostEquals(module.output()(0), -0.96355818541, 1e-9);
            NNTestAlmostEquals(module.output()(1), 0.8414709848, 1e-9);
            NNTestAlmostEquals(module.output()(2), 0.00159265291, 1e-9);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Sin<T> module;
            module.forward({ -1.3, 1.0, 3.14 });
            module.backward({ -1.3, 1.0, 3.14 }, { 2, -3, 1 });
            NNTestAlmostEquals(module.inGrad()(0), 0.53499765724, 1e-9);
            NNTestAlmostEquals(module.inGrad()(1), -1.6209069176, 1e-9);
            NNTestAlmostEquals(module.inGrad()(2), -0.99999873172, 1e-9);
        }
    }
}

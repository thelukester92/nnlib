#include "../test_identity.hpp"
#include "../test_module.hpp"
#include "nnlib/nn/identity.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Identity)
{
    NNRunAbstractTest(Module, Identity, new Identity<T>());

    NNTestMethod(forward)
    {
        Identity<T> module;
        module.forward({ -1.3, 1.0, 3.14 });
        NNTestAlmostEquals(module.output()(0), -1.3, 1e-12);
        NNTestAlmostEquals(module.output()(1), 1.0, 1e-12);
        NNTestAlmostEquals(module.output()(2), 3.14, 1e-12);
    }

    NNTestMethod(backward)
    {
        Identity<T> module;
        module.forward({ -1.3, 1.0, 3.14 });
        module.backward({ -1.3, 1.0, 3.14 }, { 2, -3, 1 });
        NNTestAlmostEquals(module.inGrad()(0), 2, 1e-12);
        NNTestAlmostEquals(module.inGrad()(1), -3, 1e-12);
        NNTestAlmostEquals(module.inGrad()(2), 1, 1e-12);
    }
}

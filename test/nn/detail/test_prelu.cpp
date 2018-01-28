#include "../test_module.hpp"
#include "../test_prelu.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/prelu.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(PReLU)
{
    NNRunAbstractTest(Module, PReLU, new PReLU<T>(5));

    NNTestMethod(operator=)
    {
        NNTestParams(const PReLU &)
        {
            PReLU<T> orig(5);
            PReLU<T> copy(10);
            copy = orig;
            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(leak)
    {
        NNTestParams(size_t)
        {
            PReLU<T> module(3);
            module.params().copy({ 1, 2, 3 });
            NNTestAlmostEquals(module.leak(0), 1, 1e-12);
            NNTestAlmostEquals(module.leak(1), 2, 1e-12);
            NNTestAlmostEquals(module.leak(2), 3, 1e-12);
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            PReLU<T> module(3);
            module.params().copy({ 0, 0.5, 2 });
            module.forward({ -1.3, 1.0, 3.14 });
            NNTestAlmostEquals(module.output()(0), 0, 1e-12);
            NNTestAlmostEquals(module.output()(1), 1.0, 1e-12);
            NNTestAlmostEquals(module.output()(2), 3.14, 1e-12);

            module.forward(Tensor<T>({ -1.3, 1.0, 3.14 }).resize(1, 3));
            NNTestAlmostEquals(module.output()(0, 0), 0, 1e-12);
            NNTestAlmostEquals(module.output()(0, 1), 1.0, 1e-12);
            NNTestAlmostEquals(module.output()(0, 2), 3.14, 1e-12);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            PReLU<T> module(3);
            module.params().copy({ 0, 0.5, 2 });
            module.forward({ -1.3, 1.0, -3.14 });
            module.backward({ -1.3, 1.0, -3.14 }, { 2, -3, 1 });
            NNTestAlmostEquals(module.inGrad()(0), 0, 1e-12);
            NNTestAlmostEquals(module.inGrad()(1), -3, 1e-12);
            NNTestAlmostEquals(module.inGrad()(2), 1, 1e-12);
            NNTestAlmostEquals(module.grad()(0), -2.6, 1e-12);
            NNTestAlmostEquals(module.grad()(1), 0, 1e-12);
            NNTestAlmostEquals(module.grad()(2), -3.14, 1e-12);

            math::fill(module.grad(), 0);
            module.forward(Tensor<T>({ -1.3, 1.0, -3.14 }).resize(1, 3));
            module.backward(Tensor<T>({ -1.3, 1.0, -3.14 }).resize(1, 3), Tensor<T>({ 2, -3, 1 }).resize(1, 3));
            NNTestAlmostEquals(module.inGrad()(0, 0), 0, 1e-12);
            NNTestAlmostEquals(module.inGrad()(0, 1), -3, 1e-12);
            NNTestAlmostEquals(module.inGrad()(0, 2), 1, 1e-12);
            NNTestAlmostEquals(module.grad()(0), -2.6, 1e-12);
            NNTestAlmostEquals(module.grad()(1), 0, 1e-12);
            NNTestAlmostEquals(module.grad()(2), -3.14, 1e-12);
        }
    }
}

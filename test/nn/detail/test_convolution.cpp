#include "../test_convolution.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/convolution.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Convolution)
{
    NNRunAbstractTest(Module, Convolution, new Convolution<T>(3, 5, 3, 3));

    NNTestMethod(Convolution)
    {
        NNTestParams(size_t, size_t, size_t, size_t, size_t, size_t, bool, bool)
        {
            Convolution<T> module(1, 2, 3, 4, 5, 6, true, false);
            NNTestEquals(module.filters(), 1);
            NNTestEquals(module.channels(), 2);
            NNTestEquals(module.kernelWidth(), 3);
            NNTestEquals(module.kernelHeight(), 4);
            NNTestEquals(module.strideX(), 5);
            NNTestEquals(module.strideY(), 6);
            NNTestEquals(module.padded(), true);
            NNTestEquals(module.interleaved(), false);

            Convolution<T> module2(6, 5, 4, 3, 2, 1, false, true);
            NNTestEquals(module2.filters(), 6);
            NNTestEquals(module2.channels(), 5);
            NNTestEquals(module2.kernelWidth(), 4);
            NNTestEquals(module2.kernelHeight(), 3);
            NNTestEquals(module2.strideX(), 2);
            NNTestEquals(module2.strideY(), 1);
            NNTestEquals(module2.padded(), false);
            NNTestEquals(module2.interleaved(), true);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Linear &)
        {
            Convolution<T> orig(1, 2, 3, 4, 5, 6, true, false);
            Convolution<T> copy(1, 2, 3, 4);
            copy = orig;

            NNTestEquals(copy.filters(), 1);
            NNTestEquals(copy.channels(), 2);
            NNTestEquals(copy.kernelWidth(), 3);
            NNTestEquals(copy.kernelHeight(), 4);
            NNTestEquals(copy.strideX(), 5);
            NNTestEquals(copy.strideY(), 6);
            NNTestEquals(copy.padded(), true);
            NNTestEquals(copy.interleaved(), false);

            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(reset)
    {
        NNTestParams()
        {
            Convolution<T> module(1, 2, 3, 4, 5, 6, true, false);
            math::fill(module.params(), 100);
            module.reset();
            forEach([&](T param)
            {
                NNTestLessThan(param, 10);
            }, module.params());
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            // todo: test padded/unpadded
            // todo: test interlaved/uninterleaved
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            // todo: test padded/unpadded
            // todo: test interlaved/uninterleaved
        }
    }
}

#include "../test_convolution.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/convolution.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Convolution)
{
    NNRunAbstractTest(Module, Convolution, new Convolution<T>(3, 5, 3));

    NNTestMethod(Convolution)
    {
        NNTestParams(size_t, size_t, const Storage<size_t> &, const Storage<size_t> &, const Storage<size_t> &, bool)
        {
            Convolution<T> module(1, 2, { 3, 4 }, { 5, 6 }, { 7, 8 }, false);
            NNTestEquals(module.kernelCount(), 1);
            NNTestEquals(module.inChannels(), 2);
            NNTestEquals(module.kernelShape(), Storage<size_t>({ 3, 4 }));
            NNTestEquals(module.stride(), Storage<size_t>({ 5, 6 }));
            NNTestEquals(module.pad(), Storage<size_t>({ 7, 8 }));
            NNTestEquals(module.interleaved(), false);

            Convolution<T> module2(8, 7, { 6, 5 }, { 4, 3 }, { 3, 2 }, true);
            NNTestEquals(module2.kernelCount(), 8);
            NNTestEquals(module2.inChannels(), 7);
            NNTestEquals(module2.kernelShape(), Storage<size_t>({ 6, 5 }));
            NNTestEquals(module2.stride(), Storage<size_t>({ 4, 3 }));
            NNTestEquals(module2.pad(), Storage<size_t>({ 3, 2 }));
            NNTestEquals(module2.interleaved(), true);
        }

        NNTestParams(size_t, size_t, size_t, size_t, size_t, bool)
        {
            Convolution<T> module(1, 2, 2, 4, 5, false);
            NNTestEquals(module.kernelCount(), 1);
            NNTestEquals(module.inChannels(), 2);
            NNTestEquals(module.kernelShape(), Storage<size_t>({ 2, 2 }));
            NNTestEquals(module.stride(), Storage<size_t>({ 4, 4 }));
            NNTestEquals(module.pad(), Storage<size_t>({ 5, 5 }));
            NNTestEquals(module.interleaved(), false);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Linear &)
        {
            Convolution<T> orig(1, 2, 2, 4, 5, false);
            Convolution<T> copy(5, 4, 3, 2, 1, true);
            copy = orig;

            NNTestEquals(copy.kernelCount(), 1);
            NNTestEquals(copy.inChannels(), 2);
            NNTestEquals(copy.kernelShape(), Storage<size_t>({ 2, 2 }));
            NNTestEquals(copy.stride(), Storage<size_t>({ 4, 4 }));
            NNTestEquals(copy.pad(), Storage<size_t>({ 5, 5 }));
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
            Convolution<T> module(1, 2, 2, 4, 5, false);
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
            Tensor<T> input, target;
            input = Tensor<T>({
                2, 2, 2, 2, 0,
                0, 0, 2, 2, 1,
                2, 0, 1, 1, 2,
                1, 0, 1, 1, 0,
                2, 1, 0, 1, 2,

                2, 1, 0, 1, 1,
                2, 2, 1, 2, 0,
                2, 2, 1, 1, 0,
                2, 2, 1, 1, 1,
                0, 0, 0, 1, 0,

                1, 2, 2, 0, 0,
                2, 0, 0, 1, 0,
                1, 0, 2, 1, 2,
                0, 2, 1, 1, 1,
                2, 2, 2, 2, 1
            }).resize(1, 3, 5, 5);

            Convolution<T> module(2, 3, 3, 2);
            module.kernels().copy({
                -1,  1, -1,  0, -1,  1,  1, -1, -1,
                 0, -1,  1,  0,  1, -1,  0, -1,  1,
                 0,  1,  0,  0, -1,  1,  0,  1,  0,

                -1,  1,  0,  0,  0,  0, -1,  1, -1,
                -1,  1,  0,  1,  1, -1, -1,  0, -1,
                 0,  1, -1,  1,  1,  1,  0,  1, -1
            });
            module.bias().copy({ 1, 0 });

            target = Tensor<T>({
                 3, -1,
                 1, -4,

                 0,  0,
                 0,  0
            }).resize(1, 2, 2, 2);

            module.forward(input);
            forEach([&](T output, T target)
            {
                NNTestAlmostEquals(output, target, 1e-12);
            }, module.output(), target);

            // todo: make module padded

            target = Tensor<T>({
                 5, -5,  3,
                -1,  0, -3,
                 1,  2, -3,

                 4, -1, -1,
                 2,  2,  0,
                 5,  5,  4
            }).resize(1, 2, 3, 3);

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

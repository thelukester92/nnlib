#include "../test_convolution.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/convolution.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Convolution)
{
/*
    NNRunAbstractTest(Module, Convolution, new Convolution<T>(3, 5, 3, 3));

    NNTestMethod(Linear)
    {
        NNTestParams(size_t, size_t)
        {
            Linear<T> module(2, 3);
            NNTestEquals(module.inputs(), 2);
            NNTestEquals(module.outputs(), 3);
            NNTestEquals(module.weights().size(), 6);
            NNTest(module.biased());
            NNTestEquals(module.bias().size(), 3);
        }

        NNTestParams(size_t, size_t, bool)
        {
            Linear<T> module(2, 3, false);
            NNTest(!module.biased());
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Linear &)
        {
            Linear<T> orig(2, 3);
            Linear<T> copy(1, 1);
            copy = orig;
            NNTestEquals(copy.inputs(), orig.inputs());
            NNTestEquals(copy.outputs(), orig.outputs());
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
            Linear<T> module(2, 3);
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
            Linear<T> module(2, 3);
            module.weights().copy({ -3, -2, 2, 3, 4, 5 });
            module.bias().copy({ -5, 7, 8862.37 });
            auto input = Tensor<T>({ -5, 10, 15, -20 }).resize(2, 2);
            auto target = Tensor<T>({ 40, 57, 8902.37, -110, -103, 8792.37 }).resize(2, 3);

            module.forward(input);
            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-12);
            }, module.output(), target);

            module.forward(input.select(0, 0));
            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-12);
            }, module.output(), target.select(0, 0));

            Linear<T> unbiased(2, 3, false);
            unbiased.weights().copy(module.weights());

            unbiased.forward(input);
            forEach([&](T unbiased, T bias, T target)
            {
                NNTestAlmostEquals(unbiased, target - bias, 1e-12);
            }, unbiased.output(), module.bias().resize(1, 3).expand(0, 2), target);

            unbiased.forward(input.select(0, 0));
            forEach([&](T unbiased, T bias, T target)
            {
                NNTestAlmostEquals(unbiased, target - bias, 1e-12);
            }, unbiased.output(), module.bias(), target.select(0, 0));
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Linear<T> module(2, 3);
            module.weights().copy({ -3, -2, 2, 3, 4, 5 });
            module.bias().copy({ -5, 7, 8862.37 });
            auto input = Tensor<T>({ -5, 10, 15, -20 }).resize(2, 2);
            auto blame = Tensor<T>({ 1, 2, 3, -4, -3, 2 }).resize(2, 3);
            auto inGrad = Tensor<T>({ -1, 26, 22, -14 }).resize(2, 2);
            auto pGrad = Tensor<T>({ -65, -55, 15, 90, 80, -10, -3, -1, 5 });

            module.forward(input);
            module.backward(input, blame);
            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-12);
            }, module.inGrad(), inGrad);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-12);
            }, module.grad(), pGrad);

            module.backward(input.select(0, 0), blame.select(0, 0));
            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-12);
            }, module.inGrad(), inGrad.select(0, 0));

            Linear<T> unbiased(2, 3, false);
            unbiased.weights().copy(module.weights());

            unbiased.forward(input);
            unbiased.backward(input, blame);
            forEach([&](T unbiased, T target)
            {
                NNTestAlmostEquals(unbiased, target, 1e-12);
            }, unbiased.inGrad(), inGrad);

            unbiased.backward(input.select(0, 0), blame.select(0, 0));
            forEach([&](T unbiased, T target)
            {
                NNTestAlmostEquals(unbiased, target, 1e-12);
            }, unbiased.inGrad(), inGrad.select(0, 0));
        }
    }
*/
}

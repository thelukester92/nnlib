#include "../test_batchnorm.hpp"
#include "../test_module.hpp"
#include "nnlib/nn/batchnorm.hpp"
#include "nnlib/math/math.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(BatchNorm)
{
    NNRunAbstractTest(Module, BatchNorm, new BatchNorm<T>(10));

    NNTestMethod(BatchNorm)
    {
        NNTestParams(size_t)
        {
            BatchNorm<T> module(5);
            NNTest(module.isTraining());
            NNTestEquals(module.inputShape()[1], 5);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(BatchNorm)
        {
            BatchNorm<T> orig(10);
            BatchNorm<T> copy(25);
            copy = orig;

            NNTestEquals(orig.inputShape(), copy.inputShape());
            NNTestEquals(orig.outputShape(), copy.outputShape());

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
            BatchNorm<T> module(10);
            math::fill(module.weights(), 10);
            math::fill(module.bias(), 5);
            module.reset();

            forEach([&](T weight)
            {
                NNTestGreaterThanOrEquals(weight, -1);
                NNTestLessThanOrEquals(weight, 1);
            }, module.weights());

            forEach([&](T bias)
            {
                NNTestAlmostEquals(bias, 0, 1e-12);
            }, module.bias());
        }
    }

    NNTestMethod(momentum)
    {
        NNTestParams(T)
        {
            BatchNorm<T> module(10);
            module.momentum(0.5);
            NNTestAlmostEquals(module.momentum(), 0.5, 1e-12);
            module.momentum(0.1);
            NNTestAlmostEquals(module.momentum(), 0.1, 1e-12);
        }
    }

    NNTestMethod(training)
    {
        NNTestParams(bool)
        {
            BatchNorm<T> module(10);
            NNTest(module.isTraining());
            module.training(false);
            NNTest(!module.isTraining());
            module.training(true);
            NNTest(module.isTraining());
        }
    }

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            BatchNorm<T> module(10);
            math::fill(module.weights(), 1);
            math::fill(module.bias(), 2);
            module.momentum(0.125);
            module.training(false);

            Serialized s;
            module.save(s);

            forEach([&](T weight)
            {
                NNTestAlmostEquals(weight, 1, 1e-12);
            }, s.get<Tensor<T>>("weights"));

            forEach([&](T bias)
            {
                NNTestAlmostEquals(bias, 2, 1e-12);
            }, s.get<Tensor<T>>("biases"));

            NNTestAlmostEquals(s.get<T>("momentum"), 0.125, 1e-12);
            NNTest(!s.get<bool>("training"));
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            auto input = Tensor<T>({ 3, 6, 9, -1, 5, 4, 12, 5, 11 }).resize(3, 3);

            BatchNorm<T> module(3);
            math::fill(module.weights(), 1);
            math::fill(module.bias(), 0);
            module.momentum(1.0);

            module.forward(input);
            for(size_t i = 0; i < 3; ++i)
            {
                NNTestAlmostEquals(math::mean(module.output().select(1, i)), 0, 1e-9);
                NNTestAlmostEquals(math::variance(module.output().select(1, i)), 1, 1e-9);
            }

            module.training(false);

            module.forward(input);
            for(size_t i = 0; i < 3; ++i)
            {
                NNTestAlmostEquals(math::mean(module.output().select(1, i)), 0, 1e-9);
                NNTestAlmostEquals(math::variance(math::scale(module.output().select(1, i), sqrt(3) / sqrt(2))), 1, 1e-9);
            }
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            auto input = Tensor<T>({ 3, 6, 9, -1, 5, 4, 12, 5, 11 }).resize(3, 3);
            auto blame = Tensor<T>({ 2, 3, 4, -2, 0, 4, 10, 2, 4 }).resize(3, 3);
            auto inGrad = Tensor<T>({ 0.03596, 0, 0, -0.02489, -2.12132, 0, -0.01106, 2.12132, 0 }).resize(3, 3);
            auto pGrad = Tensor<T>({ 14.9606, 2.82843, 0, 10, 5, 12 });

            BatchNorm<T> module(3);
            math::fill(module.weights(), 1);
            math::fill(module.bias(), 0);
            module.momentum(1.0);

            module.forward(input);
            module.backward(input, blame);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-3);
            }, module.inGrad(), inGrad);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-3);
            }, module.grad(), pGrad);

            module.training(false);
            inGrad.copy({ 0.30038, 5.19615, 1.10940, -0.30038, 0, 1.10940, 1.50188, 3.46410, 1.10940 });
            pGrad.copy({ 27.17588, 5.13783, 0, 20, 10, 24 });

            module.forward(input);
            module.backward(input, blame);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-3);
            }, module.inGrad(), inGrad);

            forEach([&](T actual, T target)
            {
                NNTestAlmostEquals(actual, target, 1e-3);
            }, module.grad(), pGrad);
        }
    }

    NNTestMethod(paramsList)
    {
        NNTestParams()
        {
            BatchNorm<T> module(10);
            math::fill(module.params(), 3.14);

            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 3.14, 1e-12);
            }, module.weights());

            forEach([&](T param)
            {
                NNTestAlmostEquals(param, 3.14, 1e-12);
            }, module.bias());
        }
    }
}

#include "../test_dropout.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/dropout.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Dropout)
{
    NNRunAbstractTest(Module, Dropout, new Dropout<T>(0.25));

    NNTestMethod(Dropout)
    {
        NNTestParams(T)
        {
            Dropout<T> module(0.7);
            NNTestAlmostEquals(module.dropProbability(), 0.7, 1e-12);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Dropout &)
        {
            Dropout<T> orig(0.7);
            Dropout<T> copy(0.3);
            copy = orig;
            NNTestAlmostEquals(copy.dropProbability(), 0.7, 1e-12);
            forEach([&](T orig, T copy)
            {
                NNTestAlmostEquals(orig, copy, 1e-12);
            }, orig.params(), copy.params());
        }
    }

    NNTestMethod(dropProbability)
    {
        NNTestParams(T)
        {
            Dropout<T> module(0.3);
            NNTestAlmostEquals(module.dropProbability(), 0.3, 1e-12);
            module.dropProbability(0.9);
            NNTestAlmostEquals(module.dropProbability(), 0.9, 1e-12);
        }
    }

    NNTestMethod(training)
    {
        NNTestParams(bool)
        {
            Dropout<T> module(0.5);
            NNTest(module.isTraining());
            module.training(false);
            NNTest(!module.isTraining());
            module.training(true);
            NNTest(module.isTraining());
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);

            T p = 0.75;
            size_t trials = 1000;

            Dropout<T> module(p);

            T sum = 0;
            auto input = math::ones<T>(4, 25);
            for(size_t i = 0; i < trials; ++i)
                sum += math::sum(module.forward(input)) / input.size();
            NNTestAlmostEquals(sum / trials, 1 - p, 0.01);

            module.training(false);
            sum = 0;
            for(size_t i = 0; i < trials; ++i)
                sum += math::sum(module.forward(input)) / input.size();
            NNTestAlmostEquals(sum / trials, 1 - p, 0.01);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);

            T p = 0.75;
            size_t trials = 1000;

            Dropout<T> module(p);

            T sum1 = 0, sum2 = 0;
            auto input = math::ones<T>(4, 25);
            for(size_t i = 0; i < trials; ++i)
            {
                sum1 += math::sum(module.forward(input)) / input.size();
                sum2 += math::sum(module.backward(input, input)) / input.size();
            }
            NNTestAlmostEquals(sum1 / trials, 1 - p, 0.01);
            NNTestAlmostEquals(sum1, sum2, 1e-12);

            module.training(false);
            sum1 = 0, sum2 = 0;
            for(size_t i = 0; i < trials; ++i)
            {
                sum1 += math::sum(module.forward(input)) / input.size();
                sum2 += math::sum(module.backward(input, input)) / input.size();
            }
            NNTestAlmostEquals(sum1 / trials, 1 - p, 0.01);
            NNTestAlmostEquals(sum1, sum2, 1e-12);
        }
    }
}

#include "../test_dropconnect.hpp"
#include "../test_module.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/nn/dropconnect.hpp"
#include "nnlib/nn/linear.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(DropConnect)
{
    NNTestMethod(DropConnect)
    {
        NNTestParams(Module *, T)
        {
            Linear<T> *linear = new Linear<T>(3, 4);
            DropConnect<T> module(linear, 0.7);
            NNTestEquals(&module.module(), linear);
            NNTestAlmostEquals(module.dropProbability(), 0.7, 1e-12);
        }
    }

    NNTestMethod(module)
    {
        NNTestParams(Module *)
        {
            Linear<T> *linear1 = new Linear<T>(3, 4);
            Linear<T> *linear2 = new Linear<T>(3, 4);
            DropConnect<T> module(linear1);
            NNTestEquals(&module.module(), linear1);
            module.module(linear2);
            NNTestEquals(&module.module(), linear2);
        }
    }

    NNTestMethod(dropProbability)
    {
        NNTestParams(T)
        {
            DropConnect<T> module(new Linear<T>(3, 4), 0.3);
            NNTestAlmostEquals(module.dropProbability(), 0.3, 1e-12);
            module.dropProbability(0.9);
            NNTestAlmostEquals(module.dropProbability(), 0.9, 1e-12);
        }
    }

    NNTestMethod(training)
    {
        NNTestParams(bool)
        {
            DropConnect<T> module(new Linear<T>(3, 4));
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
            size_t inps = 25, trials = 1000;

            auto linear = new Linear<T>(inps, 1, false);
            linear->weights().ones();

            DropConnect<T> module(linear, p);

            T sum = 0;
            auto input = Tensor<T>(4, inps).ones();
            for(size_t i = 0; i < trials; ++i)
                sum += math::sum(module.forward(input)) / module.output().size() / inps;
            NNTestAlmostEquals(sum / trials, 1 - p, 0.01);

            sum = 0;
            input.resizeDim(0, 1);
            for(size_t i = 0; i < trials; ++i)
                sum += math::sum(module.forward(input)) / module.output().size() / inps;
            NNTestAlmostEquals(sum / trials, 1 - p, 0.01);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);

            T p = 0.75;
            size_t inps = 25, trials = 1000;

            auto linear = new Linear<T>(inps, 1, false);
            linear->weights().ones();

            DropConnect<T> module(linear, p);

            T sum1 = 0, sum2 = 0;
            auto input = Tensor<T>(4, inps).ones();
            auto blame = Tensor<T>(4, 1).ones();
            for(size_t i = 0; i < trials; ++i)
            {
                sum1 += math::sum(module.forward(input)) / module.output().size() / inps;
                sum2 += math::sum(module.backward(input, blame)) / module.output().size() / inps;
            }
            NNTestAlmostEquals(sum1 / trials, 1 - p, 0.01);
            NNTestAlmostEquals(sum1, sum2, 1e-12);

            sum1 = 0, sum2 = 0;
            input.resizeDim(0, 1);
            blame.resizeDim(0, 1);
            for(size_t i = 0; i < trials; ++i)
            {
                sum1 += math::sum(module.forward(input)) / module.output().size() / inps;
                sum2 += math::sum(module.backward(input, blame)) / module.output().size() / inps;
            }
            NNTestAlmostEquals(sum1 / trials, 1 - p, 0.01);
            NNTestAlmostEquals(sum1, sum2, 1e-12);
        }
    }
}

#include "../test_adam.hpp"
#include "../test_optimizer.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
#include "nnlib/nn/linear.hpp"
#include "nnlib/opt/adam.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Adam)
{
    Linear<T> model(2, 3);
    MSE<T> critic;
    NNRunAbstractTest(Optimizer, Adam, new Adam<T>(model, critic));

    NNTestMethod(reset)
    {
        NNTestParams()
        {
            RandomEngine::sharedEngine().seed(0);

            Linear<T> model(2, 3);
            MSE<T> critic;
            Adam<T> opt(model, critic);

            auto inputs = math::rand(Tensor<T>(2));
            auto target = math::rand(Tensor<T>(3));

            auto before = model.params().copy();
            for(size_t i = 0; i < 100; ++i)
                opt.step(inputs, target);
            auto after = model.params().copy();

            model.params().copy(before);
            opt.reset();
            for(size_t i = 0; i < 100; ++i)
                opt.step(inputs, target);

            forEach([&](T first, T second)
            {
                NNTestAlmostEquals(first, second, 1e-12);
            }, after, model.params());
        }
    }

    NNTestMethod(learningRate)
    {
        NNTestParams(T)
        {
            RandomEngine::sharedEngine().seed(0);

            Linear<T> model(2, 3);
            MSE<T> critic;
            Adam<T> opt(model, critic);

            auto inputs = math::rand(Tensor<T>(2));
            auto target = math::rand(Tensor<T>(3));

            opt.learningRate(0.1);
            NNTestAlmostEquals(opt.learningRate(), 0.1, 1e-12);

            auto before = model.params().copy();
            opt.step(inputs, target);
            T dist1 = sqrt(math::sum(math::square(before - model.params())));

            model.params().copy(before);
            opt.reset();
            opt.learningRate(1);
            opt.step(inputs, target);
            T dist2 = sqrt(math::sum(math::square(before - model.params())));

            NNTestAlmostEquals(dist1, 0.1 * dist2, 1e-12);
        }
    }

    NNTestMethod(beta1)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1);
            model.params().fill(1);
            MSE<T> critic;
            Adam<T> opt(model, critic);
            opt.beta1(0.5);
            NNTestAlmostEquals(opt.beta1(), 0.5, 1e-12);
        }
    }

    NNTestMethod(beta2)
    {
        NNTestParams(T)
        {
            Linear<T> model(1, 1);
            model.params().fill(1);
            MSE<T> critic;
            Adam<T> opt(model, critic);
            opt.beta2(0.5);
            NNTestAlmostEquals(opt.beta2(), 0.5, 1e-12);
        }
    }
}

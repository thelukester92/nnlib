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
        Linear<T> model(2, 3);
        MSE<T> critic;
        Adam<T> opt(model, critic);

        RandomEngine::sharedEngine().seed(0);
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

    NNTestMethod(learningRate)
    {

    }

    NNTestMethod(beta1)
    {

    }

    NNTestMethod(beta2)
    {

    }
}

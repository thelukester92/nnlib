#include "../test_optimizer.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Optimizer, Optimizer<T>)
{
    NNTestMethod(params)
    {
        NNTestParams()
        {
            NNTestEquals(&nnImpl.params(), &nnImpl.model().params());
        }
    }

    NNTestMethod(grad)
    {
        NNTestParams()
        {
            NNTestEquals(&nnImpl.grad(), &nnImpl.model().grad());
        }
    }

    NNTestMethod(learningRate)
    {
        NNTestParams(T)
        {
            RandomEngine::sharedEngine().seed(0);

            auto inputs = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));
            auto target = math::rand(Tensor<T>(nnImpl.model().outputShape(), true));

            nnImpl.learningRate(0.1);
            NNTestAlmostEquals(nnImpl.learningRate(), 0.1, 1e-12);

            auto before = nnImpl.params().copy();
            nnImpl.step(inputs, target);
            T dist1 = sqrt(math::sum(math::square(before - nnImpl.params())));

            nnImpl.params().copy(before);
            nnImpl.reset();
            nnImpl.learningRate(1);
            NNTestAlmostEquals(nnImpl.learningRate(), 1, 1e-12);
            nnImpl.step(inputs, target);
            T dist2 = sqrt(math::sum(math::square(before - nnImpl.params())));

            NNTestAlmostEquals(dist1, 0.1 * dist2, 1e-12);
        }
    }

    NNTestMethod(evaluate)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);

            auto inputs = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));
            auto target = math::rand(Tensor<T>(nnImpl.model().outputShape(), true));

            nnImpl.reset();
            T direct = nnImpl.critic().forward(nnImpl.model().forward(inputs), target);
            nnImpl.reset();
            T convenience = nnImpl.evaluate(inputs, target);

            NNTestAlmostEquals(direct, convenience, 1e-12);
        }
    }

    NNTestMethod(reset)
    {
        NNTestParams()
        {
            RandomEngine::sharedEngine().seed(0);

            auto inputs = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));
            auto target = math::rand(Tensor<T>(nnImpl.model().outputShape(), true));

            auto before = nnImpl.params().copy();
            for(size_t i = 0; i < 100; ++i)
                nnImpl.step(inputs, target);
            auto after = nnImpl.params().copy();

            nnImpl.params().copy(before);
            nnImpl.reset();
            for(size_t i = 0; i < 100; ++i)
                nnImpl.step(inputs, target);

            forEach([&](T first, T second)
            {
                NNTestAlmostEquals(first, second, 1e-12);
            }, after, nnImpl.params());
        }
    }

    NNTestMethod(step)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);
            math::rand(nnImpl.params());

            auto inputs = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));
            auto target = math::rand(Tensor<T>(nnImpl.model().outputShape(), true));

            T errBefore = nnImpl.evaluate(inputs, target);
            nnImpl.step(inputs, target);
            T errAfter = nnImpl.evaluate(inputs, target);

            NNTestLessThan(errAfter, errBefore);
        }
    }
}

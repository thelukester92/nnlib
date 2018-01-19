#include "../test_optimizer.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Optimizer, Optimizer<T>)
{
    NNTestMethod(reset)
    {
        NNTestParams()
        {
            RandomEngine::sharedEngine().seed(0);

            auto inputs = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));
            auto target = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));

            auto before = nnImpl.model().params().copy();
            for(size_t i = 0; i < 100; ++i)
                nnImpl.step(inputs, target);
            auto after = nnImpl.model().params().copy();

            nnImpl.model().params().copy(before);
            nnImpl.reset();
            for(size_t i = 0; i < 100; ++i)
                nnImpl.step(inputs, target);

            forEach([&](T first, T second)
            {
                NNTestAlmostEquals(first, second, 1e-12);
            }, after, nnImpl.model().params());
        }
    }

    NNTestMethod(step)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);

            auto inputs = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));
            auto target = math::rand(Tensor<T>(nnImpl.model().outputShape(), true));

            T errBefore = nnImpl.evaluate(inputs, target);
            nnImpl.step(inputs, target);
            T errAfter = nnImpl.evaluate(inputs, target);

            NNTestLessThan(errAfter, errBefore);
        }
    }
}

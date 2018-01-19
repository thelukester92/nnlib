#include "../test_optimizer.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Optimizer, Optimizer<T>)
{
    NNTestMethod(step)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            RandomEngine::sharedEngine().seed(0);

            auto inputs = math::rand(Tensor<T>(nnImpl.model().inputShape(), true));
            auto target = math::rand(Tensor<T>(nnImpl.model().outputShape(), true));

            T errBefore = nnImpl.critic().forward(nnImpl.model().forward(inputs), target);
            nnImpl.step(inputs, target);
            T errAfter = nnImpl.critic().forward(nnImpl.model().forward(inputs), target);

            NNTestLessThan(errAfter, errBefore);
        }
    }
}

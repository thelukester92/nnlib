#include "../test_critic.hpp"
#include "../test_criticsequencer.hpp"
#include "nnlib/critics/criticsequencer.hpp"
#include "nnlib/critics/mse.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(CriticSequencer)
{
    NNRunAbstractTest(Critic, CriticSequencer, new CriticSequencer<T>(new MSE<T>()));

    NNTestMethod(CriticSequencer)
    {
        NNTestParams(Critic *)
        {
            MSE<T> *innerCritic = new MSE<T>(false);
            CriticSequencer<T> critic(innerCritic);
            NNTestEquals(&critic.critic(), innerCritic);
        }
    }

    NNTestMethod(critic)
    {
        NNTestParams()
        {
            MSE<T> *innerCritic = new MSE<T>(false);
            CriticSequencer<T> critic(innerCritic);
            NNTestEquals(&critic.critic(), innerCritic);
        }

        NNTestParams(Critic *)
        {
            MSE<T> *innerCritic = new MSE<T>(false);
            CriticSequencer<T> critic(nullptr);
            critic.critic(innerCritic);
            NNTestEquals(&critic.critic(), innerCritic);
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            CriticSequencer<T> critic(new MSE<T>(false));
            Tensor<T> inputs = Tensor<T>({ 1, 2, 3 }).resize(3, 1, 1);
            Tensor<T> target = Tensor<T>({ 0, 3, 5 }).resize(3, 1, 1);
            NNTestAlmostEquals(critic.forward(inputs, target), 6, 1e-12);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            CriticSequencer<T> critic(new MSE<T>(false));
            Tensor<T> inputs = Tensor<T>({ 1, 2, 3 }).resize(3, 1, 1);
            Tensor<T> target = Tensor<T>({ 0, 3, 5 }).resize(3, 1, 1);
            Tensor<T> inGrad = critic.backward(inputs, target);
            NNTestAlmostEquals(inGrad(0, 0, 0),  2, 1e-12);
            NNTestAlmostEquals(inGrad(1, 0, 0), -2, 1e-12);
            NNTestAlmostEquals(inGrad(2, 0, 0), -4, 1e-12);
        }
    }
}

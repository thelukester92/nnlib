#include "../test_critic.hpp"
#include "../test_mse.hpp"
#include "nnlib/critics/mse.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(MSE)
{
    NNRunAbstractTest(Critic, MSE, new MSE<T>());

    NNTestMethod(MSE)
    {
        NNTestParams()
        {
            MSE<T> critic;
            NNTestEquals(critic.average(), true);
        }

        NNTestParams(bool)
        {
            MSE<T> critic(false);
            NNTestEquals(critic.average(), false);
        }
    }

    NNTestMethod(average)
    {
        NNTestParams()
        {
            MSE<T> critic;
            NNTestEquals(critic.average(), true);
        }

        NNTestParams(bool)
        {
            MSE<T> critic;
            NNTestEquals(&critic.average(false), &critic);
            NNTestEquals(critic.average(), false);
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            MSE<T> critic(false);
            Tensor<T> inputs = Tensor<T>({ 1, 2, 3 });
            Tensor<T> target = Tensor<T>({ 0, 3, 5 });
            NNTestAlmostEquals(critic.forward(inputs, target), 6, 1e-12);
            critic.average(true);
            NNTestAlmostEquals(critic.forward(inputs, target), 2, 1e-12);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            MSE<T> critic(false);
            Tensor<T> inputs = Tensor<T>({ 1, 2, 3 });
            Tensor<T> target = Tensor<T>({ 0, 3, 5 });
            Tensor<T> inGrad = critic.backward(inputs, target);
            NNTestAlmostEquals(inGrad(0),  2, 1e-12);
            NNTestAlmostEquals(inGrad(1), -2, 1e-12);
            NNTestAlmostEquals(inGrad(2), -4, 1e-12);
            critic.average(true);
            inGrad = critic.backward(inputs, target);
            NNTestAlmostEquals(inGrad(0),  2.0 / 3.0, 1e-12);
            NNTestAlmostEquals(inGrad(1), -2.0 / 3.0, 1e-12);
            NNTestAlmostEquals(inGrad(2), -4.0 / 3.0, 1e-12);
        }
    }
}

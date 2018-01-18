#include "../test_critic.hpp"
#include "../test_nll.hpp"
#include "nnlib/critics/nll.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(NLL)
{
    NNTestMethod(NLL)
    {
        NNTestParams()
        {
            NLL<T> critic;
            NNTestEquals(critic.average(), true);
        }

        NNTestParams(bool)
        {
            NLL<T> critic(false);
            NNTestEquals(critic.average(), false);
        }
    }

    NNTestMethod(average)
    {
        NNTestParams()
        {
            NLL<T> critic;
            NNTestEquals(critic.average(), true);
        }

        NNTestParams(bool)
        {
            NLL<T> critic;
            NNTestEquals(&critic.average(false), &critic);
            NNTestEquals(critic.average(), false);
        }
    }

    NNTestMethod(misclassifications)
    {
        NNTestParams(const Tensor<T> &, const Tensor<T> &)
        {
            NLL<T> critic(false);
            Tensor<T> inputs = Tensor<T>({
                -0.1, -1.0, -1.0, -1.0,
                -0.7, -0.2, -0.7, -0.7,
                -0.6, -0.6, -0.3, -0.6
            }).resize(3, 4);
            Tensor<T> target = Tensor<T>({ 0, 1, 3 }).resize(3, 1);
            NNTestEquals(critic.misclassifications(inputs, target), 1);
        }
    }

    NNTestMethod(forward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            NLL<T> critic(false);
            Tensor<T> inputs = Tensor<T>({
                -0.1, -1.0, -1.0, -1.0,
                -0.7, -0.2, -0.7, -0.7,
                -0.6, -0.6, -0.3, -0.6
            }).resize(3, 4);
            Tensor<T> target = Tensor<T>({ 0, 1, 3 }).resize(3, 1);
            NNTestAlmostEquals(critic.forward(inputs, target), 0.9, 1e-12);
            critic.average(true);
            NNTestAlmostEquals(critic.forward(inputs, target), 0.075, 1e-12);
        }
    }

    NNTestMethod(backward)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            NLL<T> critic(false);
            Tensor<T> inputs = Tensor<T>({
                -0.1, -1.0, -1.0, -1.0,
                -0.7, -0.2, -0.7, -0.7,
                -0.6, -0.6, -0.3, -0.6
            }).resize(3, 4);
            Tensor<T> target = Tensor<T>({ 0, 1, 3 }).resize(3, 1);

            Tensor<T> inGrad = critic.backward(inputs, target);
            for(size_t i = 0; i < 3; ++i)
            {
                for(size_t j = 0; j < 4; ++j)
                {
                    if(target(i, 0) == j)
                    {
                        NNTestAlmostEquals(inGrad(i, j), -1, 1e-12);
                    }
                    else
                    {
                        NNTestAlmostEquals(inGrad(i, j), 0, 1e-12);
                    }
                }
            }

            critic.average(true);
            inGrad = critic.backward(inputs, target);
            for(size_t i = 0; i < 3; ++i)
            {
                for(size_t j = 0; j < 4; ++j)
                {
                    if(target(i, 0) == j)
                    {
                        NNTestAlmostEquals(inGrad(i, j), -1.0 / 12.0, 1e-12);
                    }
                    else
                    {
                        NNTestAlmostEquals(inGrad(i, j), 0, 1e-12);
                    }
                }
            }
        }
    }
}

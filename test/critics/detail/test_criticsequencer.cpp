#include "../test_critic.hpp"
#include "../test_criticsequencer.hpp"
#include "nnlib/critics/criticsequencer.hpp"
#include "nnlib/critics/mse.hpp"
#include "nnlib/math/math.hpp"
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

void TestCriticSequencer()
{
    Storage<size_t> shape = { 5, 1, 1 };
    Tensor<NN_REAL_T> inp = Tensor<NN_REAL_T>({  1,  2,  3,  4,  5 }).resize(shape);
    Tensor<NN_REAL_T> tgt = Tensor<NN_REAL_T>({  2,  4,  6,  8,  0 }).resize(shape);
    Tensor<NN_REAL_T> sqd = Tensor<NN_REAL_T>({  1,  4,  9, 16, 25 }).resize(shape);
    Tensor<NN_REAL_T> dif = Tensor<NN_REAL_T>({ -2, -4, -6, -8, 10 }).resize(shape);
    MSE<NN_REAL_T> *innerCritic = new MSE<NN_REAL_T>(false);
    CriticSequencer<NN_REAL_T> critic(innerCritic);

    double mse = critic.forward(inp, tgt);
    NNAssertAlmostEquals(mse, math::sum(sqd), 1e-12, "CriticSequencer::forward with no average failed!");

    critic.backward(inp, tgt);
    NNAssert(math::sum(math::square(critic.inGrad().reshape(5, 1) - dif.reshape(5, 1))) < 1e-12, "CriticSequencer::backward failed!");
}

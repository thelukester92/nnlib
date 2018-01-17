#include "../test_critic.hpp"
#include "nnlib/math/math.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestAbstractClassImpl(Critic, Critic<T>)
{
    NNTestMethod(inGrad)
    {
        NNTestParams()
        {
            Tensor<T> t = math::rand(nnImpl.inGrad().copy());
            Tensor<T> s = math::rand(nnImpl.inGrad().copy());
            Tensor<T> u = nnImpl.backward(t, s);
            NNTestEquals(u.ptr(), nnImpl.inGrad().ptr());
            const auto &cImpl = nnImpl;
            NNTestEquals(u.ptr(), cImpl.inGrad().ptr());
        }
    }
}

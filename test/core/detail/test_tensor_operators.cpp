#include "../test_tensor_operators.hpp"
#include "nnlib/core/detail/tensor_operators.hpp"
#include <sstream>
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(TensorOperators)
{
    NNTestMethod(operator<<)
    {
        NNTestParams(std::ostream &, const Tensor &)
        {
            Tensor<T> t(1);
            std::stringstream ss;
            NNTestEquals(&(ss << t), &ss);
            NNTestEquals(ss.str(), std::string("0.00000\n[ Tensor of dimension 1 ]"));
        }
    }

    NNTestMethod(operator+=)
    {
        NNTestParams(Tensor &, const Tensor &)
        {
            Tensor<T> t(1), s({ 2 });
            NNTestEquals(&(t += s), &t);
            NNTestAlmostEquals(t(0), 2, 1e-12);
        }
    }

    NNTestMethod(operator+)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Tensor<T> t(1), s({ 2 });
            Tensor<T> u = t + s;
            NNTestEquals(u.dims(), 1);
            NNTestEquals(u.size(), 1);
            NNTestAlmostEquals(u(0), 2, 1e-12);
            NNTestAlmostEquals(t(0), 0, 1e-12);
            NNTest(!u.sharedWith(t));
            NNTest(!u.sharedWith(s));
        }
    }

    NNTestMethod(operator-=)
    {
        NNTestParams(Tensor &, const Tensor &)
        {
            Tensor<T> t(1), s({ 2 });
            NNTestEquals(&(t -= s), &t);
            NNTestAlmostEquals(t(0), -2, 1e-12);
        }
    }

    NNTestMethod(operator-)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Tensor<T> t(1), s({ 2 });
            Tensor<T> u = t - s;
            NNTestEquals(u.dims(), 1);
            NNTestEquals(u.size(), 1);
            NNTestAlmostEquals(u(0), -2, 1e-12);
            NNTestAlmostEquals(t(0), 0, 1e-12);
            NNTest(!u.sharedWith(t));
            NNTest(!u.sharedWith(s));
        }
    }

    NNTestMethod(operator*=)
    {
        NNTestParams(Tensor &, T)
        {
            Tensor<T> t({ 2 });
            NNTestEquals(&(t *= 3.14), &t);
            NNTestAlmostEquals(t(0), 6.28, 1e-12);
        }
    }

    NNTestMethod(operator*)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Tensor<T> t({ 2 });
            Tensor<T> u = t * 3.14;
            NNTestEquals(u.dims(), 1);
            NNTestEquals(u.size(), 1);
            NNTestAlmostEquals(u(0), 6.28, 1e-12);
            NNTestAlmostEquals(t(0), 2, 1e-12);
            NNTest(!u.sharedWith(t));
        }
    }
}

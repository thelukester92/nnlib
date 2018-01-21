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
            Tensor<T> t = { 1, -2, 3.14159265, 123456789, 0, 42 };
            std::stringstream ss;
            NNTestEquals(&(ss << t), &ss);
            NNTest(ss.str() == "1\n-2\n3.14159265\n123456789\n0\n42\n[ Tensor of dimension 6 ]");
            t.resize(2, 3);
            std::stringstream ss2;
            ss2 << t;
            NNTest(ss2.str() == "1            -2  3.14159  \n1.23457e+08  0   42       \n[ Tensor of dimension 2 x 3 ]");
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
        NNTestParams(const Tensor &, T)
        {
            Tensor<T> t({ 2 });
            Tensor<T> u = t * 3.14;
            NNTestEquals(u.dims(), 1);
            NNTestEquals(u.size(), 1);
            NNTestAlmostEquals(u(0), 6.28, 1e-12);
            NNTestAlmostEquals(t(0), 2, 1e-12);
            NNTest(!u.sharedWith(t));
        }

        NNTestParams(T, const Tensor &)
        {
            Tensor<T> t({ 2 });
            Tensor<T> u = 3.14 * t;
            NNTestEquals(u.dims(), 1);
            NNTestEquals(u.size(), 1);
            NNTestAlmostEquals(u(0), 6.28, 1e-12);
            NNTestAlmostEquals(t(0), 2, 1e-12);
            NNTest(!u.sharedWith(t));
        }
    }

    NNTestMethod(operator/=)
    {
        NNTestParams(Tensor &, T)
        {
            Tensor<T> t({ 3.14 });
            NNTestEquals(&(t /= 3.14), &t);
            NNTestAlmostEquals(t(0), 1, 1e-12);
        }
    }

    NNTestMethod(operator/)
    {
        NNTestParams(const Tensor &, const Tensor &)
        {
            Tensor<T> t({ 6.28 });
            Tensor<T> u = t / 3.14;
            NNTestEquals(u.dims(), 1);
            NNTestEquals(u.size(), 1);
            NNTestAlmostEquals(u(0), 2, 1e-12);
            NNTestAlmostEquals(t(0), 6.28, 1e-12);
            NNTest(!u.sharedWith(t));
        }
    }
}

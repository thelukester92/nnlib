#include "../test_math.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
using namespace nnlib;
using namespace nnlib::math;
using T = NN_REAL_T;

NNTestClassImpl(Math)
{
    NNTestMethod(min)
    {
        NNTestParams(const Tensor &)
        {
            Tensor<T> x({ 8, -6, 7, 5, 3, 0, 9, 3.14159 });
            NNTestAlmostEquals(min(x), -6, 1e-12);
        }
    }

    NNTestMethod(max)
    {
        NNTestParams(const Tensor &)
        {
            Tensor<T> x({ 8, -6, 7, 5, 3, 0, 9, 3.14159 });
            NNTestAlmostEquals(max(x), 9, 1e-12);
        }
    }

    NNTestMethod(sum)
    {
        NNTestParams(const Tensor &)
        {
            Tensor<T> x({ 8, -6, 7, 5, 3, 0, 9, 3.14159 });
            NNTestAlmostEquals(sum(x), 29.14159, 1e-12);
        }
    }

    NNTestMethod(mean)
    {
        NNTestParams(const Tensor &)
        {
            Tensor<T> x({ 8, -6, 7, 5, 3, 0, 9, 3.14159 });
            NNTestAlmostEquals(mean(x), 3.64269875, 1e-12);
        }
    }

    NNTestMethod(variance)
    {
        NNTestParams(const Tensor &)
        {
            Tensor<T> x({ 8, -6, 7, 5, 3, 0, 9, 3.14159 });
            NNTestAlmostEquals(variance(x), 20.964444282761, 1e-12);
        }

        NNTestParams(const Tensor &, bool)
        {
            Tensor<T> x({ 8, -6, 7, 5, 3, 0, 9, 3.14159 });
            NNTestAlmostEquals(variance(x, true), 23.959364894584, 1e-12);
        }
    }

    NNTestMethod(normalize)
    {
        NNTestParams(Tensor &, T, T)
        {
            Tensor<T> x({ -2, 0, 8, 3 });
            normalize(x, -1, 4);
            NNTestAlmostEquals(x(0), -1, 1e-12);
            NNTestAlmostEquals(x(1), 0, 1e-12);
            NNTestAlmostEquals(x(2), 4, 1e-12);
            NNTestAlmostEquals(x(3), 1.5, 1e-12);
        }

        NNTestParams(Tensor &&, T, T)
        {
            Tensor<T> x = normalize(Tensor<T>({ -2, 0, 8, 3 }), -1, 4);
            NNTestAlmostEquals(x(0), -1, 1e-12);
            NNTestAlmostEquals(x(1), 0, 1e-12);
            NNTestAlmostEquals(x(2), 4, 1e-12);
            NNTestAlmostEquals(x(3), 1.5, 1e-12);
        }
    }

    NNTestMethod(clip)
    {
        NNTestParams(Tensor &, T, T)
        {
            Tensor<T> x({ -2, 0, 8, 3 });
            clip(x, -1, 4);
            NNTestAlmostEquals(x(0), -1, 1e-12);
            NNTestAlmostEquals(x(1), 0, 1e-12);
            NNTestAlmostEquals(x(2), 4, 1e-12);
            NNTestAlmostEquals(x(3), 3, 1e-12);
        }

        NNTestParams(Tensor &&, T, T)
        {
            Tensor<T> x = clip(Tensor<T>({ -2, 0, 8, 3 }), -1, 4);
            NNTestAlmostEquals(x(0), -1, 1e-12);
            NNTestAlmostEquals(x(1), 0, 1e-12);
            NNTestAlmostEquals(x(2), 4, 1e-12);
            NNTestAlmostEquals(x(3), 3, 1e-12);
        }
    }

    NNTestMethod(square)
    {
        NNTestParams(Tensor &)
        {
            Tensor<T> x({ -2, 0, 1, 6 });
            square(x);
            NNTestAlmostEquals(x(0), 4, 1e-12);
            NNTestAlmostEquals(x(1), 0, 1e-12);
            NNTestAlmostEquals(x(2), 1, 1e-12);
            NNTestAlmostEquals(x(3), 36, 1e-12);
        }

        NNTestParams(Tensor &&)
        {
            Tensor<T> x = square(Tensor<T>({ -2, 0, 1, 6 }));
            NNTestAlmostEquals(x(0), 4, 1e-12);
            NNTestAlmostEquals(x(1), 0, 1e-12);
            NNTestAlmostEquals(x(2), 1, 1e-12);
            NNTestAlmostEquals(x(3), 36, 1e-12);
        }
    }

    NNTestMethod(rand)
    {
        NNTestParams(Tensor &, T, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x(1000);
            rand(x, -1, 1);
            NNTestAlmostEquals(mean(x), 0, 0.5);
            NNTestAlmostEquals(variance(x), 0.3333, 0.5);
        }

        NNTestParams(Tensor &&, T, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x = rand(Tensor<T>(1000), -1, 1);
            NNTestAlmostEquals(mean(x), 0, 0.5);
            NNTestAlmostEquals(variance(x), 0.3333, 0.5);
        }
    }

    NNTestMethod(randn)
    {
        NNTestParams(Tensor &, T, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x(1000);
            randn(x, -1, 3.14);
            NNTestAlmostEquals(mean(x), -1, 0.5);
            NNTestAlmostEquals(variance(x), 9.8596, 0.5);
        }

        NNTestParams(Tensor &&, T, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x = randn(Tensor<T>(1000), -1, 3.14);
            NNTestAlmostEquals(mean(x), -1, 0.5);
            NNTestAlmostEquals(variance(x), 9.8596, 0.5);
        }

        NNTestParams(Tensor &, T, T, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x(1000);
            randn(x, -1, 3.14, 3);
            NNTestAlmostEquals(mean(x), -1, 0.5);
            NNTestGreaterThanOrEquals(min(x), -4);
            NNTestLessThanOrEquals(max(x), 2);
        }

        NNTestParams(Tensor &&, T, T, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x = randn(Tensor<T>(1000), -1, 3.14, 3);
            NNTestAlmostEquals(mean(x), -1, 0.5);
            NNTestGreaterThanOrEquals(min(x), -4);
            NNTestLessThanOrEquals(max(x), 2);
        }
    }

    NNTestMethod(bernoulli)
    {
        NNTestParams(Tensor &, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x(1000);
            bernoulli(x, 0.25);
            size_t n = 0;
            forEach([&](T x)
            {
                if(x > 0.5)
                    ++n;
            }, x);
            NNTestAlmostEquals(n, 0.25 * x.size(), 50);
        }

        NNTestParams(Tensor &&, T)
        {
            RandomEngine::sharedEngine().seed(0);
            Tensor<T> x = bernoulli(Tensor<T>(1000), 0.25);
            size_t n = 0;
            forEach([&](T x)
            {
                if(x > 0.5)
                    ++n;
            }, x);
            NNTestAlmostEquals(n, 0.25 * x.size(), 50);
        }
    }

    NNTestMethod(sum)
    {
        NNTestParams(const Tensor &, Tensor &, size_t)
        {
            Tensor<T> x = Tensor<T>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
            Tensor<T> y = Tensor<T>({ 5, 7, 9 });
            Tensor<T> z(3);
            sum(x, z, 0);
            forEach([&](T y, T z)
            {
                NNTestAlmostEquals(y, z, 1e-12);
            }, y, z);
            y = Tensor<T>({ 6, 15 });
            z.resize(2);
            sum(x, z, 1);
            forEach([&](T y, T z)
            {
                NNTestAlmostEquals(y, z, 1e-12);
            }, y, z);
        }

        NNTestParams(const Tensor &, Tensor &&, size_t)
        {
            Tensor<T> x = Tensor<T>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
            Tensor<T> y = Tensor<T>({ 5, 7, 9 });
            Tensor<T> z = sum(x, Tensor<T>(3), 0);
            forEach([&](T y, T z)
            {
                NNTestAlmostEquals(y, z, 1e-12);
            }, y, z);
        }
    }

    NNTestMethod(pointwiseProduct)
    {
        NNTestParams(const Tensor &, Tensor &)
        {
            Tensor<T> x = Tensor<T>({ 1, 2, 3 });
            Tensor<T> y = Tensor<T>({ 4, 5, 6 });
            pointwiseProduct(x, y);
            NNTestAlmostEquals(y(0), 4, 1e-12);
            NNTestAlmostEquals(y(1), 10, 1e-12);
            NNTestAlmostEquals(y(2), 18, 1e-12);
        }

        NNTestParams(const Tensor &, Tensor &&)
        {
            Tensor<T> x = Tensor<T>({ 1, 2, 3 });
            Tensor<T> y = pointwiseProduct(x, Tensor<T>({ 4, 5, 6 }));
            NNTestAlmostEquals(y(0), 4, 1e-12);
            NNTestAlmostEquals(y(1), 10, 1e-12);
            NNTestAlmostEquals(y(2), 18, 1e-12);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &)
        {
            Tensor<T> x = Tensor<T>({ 1, 2, 3 });
            Tensor<T> y = Tensor<T>({ 4, 5, 6 });
            Tensor<T> z(3);
            pointwiseProduct(x, y, z);
            NNTestAlmostEquals(z(0), 4, 1e-12);
            NNTestAlmostEquals(z(1), 10, 1e-12);
            NNTestAlmostEquals(z(2), 18, 1e-12);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &&)
        {
            Tensor<T> x = Tensor<T>({ 1, 2, 3 });
            Tensor<T> y = Tensor<T>({ 4, 5, 6 });
            Tensor<T> z = pointwiseProduct(x, y, Tensor<T>(3));
            NNTestAlmostEquals(z(0), 4, 1e-12);
            NNTestAlmostEquals(z(1), 10, 1e-12);
            NNTestAlmostEquals(z(2), 18, 1e-12);
        }
    }
}

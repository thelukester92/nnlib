#include "../test_random.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/math/random.hpp"
using namespace nnlib;
using namespace nnlib::math;
using T = NN_REAL_T;

NNTestClassImpl(RandomEngine)
{
    NNTestMethod(sharedEngine)
    {
        NNTestParams()
        {
            RandomEngine *e = &RandomEngine::sharedEngine();
            RandomEngine *f = &RandomEngine::sharedEngine();
            NNTestEquals(e, f);
        }
    }

    NNTestMethod(RandomEngine)
    {
        NNTestParams()
        {
            RandomEngine e;
            RandomEngine &f = RandomEngine::sharedEngine();
            NNTestNotEquals(&e.engine(), &f.engine());
        }

        NNTestParams(size_t)
        {
            RandomEngine e(0);
            RandomEngine f(0);
            Random<size_t> r(&e);
            Random<size_t> s(&f);
            for(size_t i = 0; i < 100; ++i)
                NNTestEquals(r.uniform(), s.uniform());
        }
    }

    NNTestMethod(seed)
    {
        NNTestParams(size_t)
        {
            RandomEngine e, f;
            e.seed(0);
            f.seed(0);
            Random<size_t> r(&e);
            Random<size_t> s(&f);
            for(size_t i = 0; i < 100; ++i)
                NNTestEquals(r.uniform(), s.uniform());
        }
    }

    NNTestMethod(engine)
    {
        NNTestParams()
        {
            RandomEngine e;
            RandomEngine &f = RandomEngine::sharedEngine();
            NNTestNotEquals(&e.engine(), &f.engine());
        }
    }
}

NNTestClassImpl(Random)
{
    NNTestMethod(sharedRandom)
    {
        NNTestParams()
        {
            Random<T> *r = &Random<T>::sharedRandom();
            Random<T> *s = &Random<T>::sharedRandom();
            NNTestEquals(r, s);
        }
    }

    NNTestMethod(Random)
    {
        NNTestParams()
        {
            Random<T> r;
            Random<T> s;
            for(size_t i = 0; i < 100; ++i)
                NNTestEquals(r.uniform(), s.uniform());
        }

        NNTestParams(RandomEngine *)
        {
            Random<T> r(new RandomEngine(0));
            Random<T> s(new RandomEngine(0));
            for(size_t i = 0; i < 100; ++i)
                NNTestEquals(r.uniform(), s.uniform());
        }
    }

    NNTestMethod(uniform)
    {
        NNTestParams(size_t)
        {
            Random<size_t> r;
            Tensor<T> x(1000);
            forEach([&](T &x)
            {
                x = r.uniform(101);
            }, x);
            NNTestAlmostEquals(mean(x), 50, 10);
            NNTestAlmostEquals(variance(x), 833.3333, 100);
        }

        NNTestParams(size_t, size_t)
        {
            Random<size_t> r;
            Tensor<T> x(1000);
            forEach([&](T &x)
            {
                x = r.uniform(80, 121);
            }, x);
            NNTestAlmostEquals(mean(x), 100, 10);
            NNTestAlmostEquals(variance(x), 133.3333, 40);
        }
    }
}

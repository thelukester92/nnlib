#include "../test_storage.hpp"
#include "nnlib/core/storage.hpp"
#include "nnlib/serialization/serialized.hpp"
using namespace nnlib;

NNTestClassImpl(Storage)
{
    NNTestMethod(Storage)
    {
        NNTestParams()
        {
            Storage<int> s;
            NNTestEquals(s.size(), 0);
        }

        NNTestParams(size_t, T)
        {
            Storage<int> s(5, 42);
            NNTestEquals(s.size(), 5);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(s[i], 42);
        }

        NNTestParams(const Storage &)
        {
            Storage<int> s(5, 42);
            Storage<int> t(s);
            NNTestEquals(t.size(), 5);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(t[i], 42);
        }

        NNTestParams(Storage &&)
        {
            Storage<int> s(5, 42);
            Storage<int> t(std::move(s));
            NNTestEquals(t.size(), 5);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(t[i], 42);
        }

        NNTestParams(const std::initializer_list &)
        {
            Storage<int> s({ 0, 1, 2, 3, 4, 5 });
            NNTestEquals(s.size(), 6);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s[i], i);
        }

        NNTestParams(const Serialized &)
        {
            Storage<int> s(5, 42);
            Storage<int> t((Serialized(s)));
            NNTestEquals(t.size(), 5);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(t[i], 42);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Storage &)
        {
            Storage<int> s(5, 42), t;
            t = s;
            NNTestEquals(t.size(), 5);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(t[i], 42);
        }

        NNTestParams(const std::initializer_list<T> &)
        {
            Storage<int> s;
            s = { 0, 1, 2, 3, 4, 5 };
            NNTestEquals(s.size(), 6);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s[i], i);
        }
    }

    NNTestMethod(resize)
    {
        NNTestParams(size_t, const T &)
        {
            Storage<int> s;
            s.resize(5, 42);
            NNTestEquals(s.size(), 5);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(s[i], 42);
        }
    }

    NNTestMethod(reserve)
    {
        NNTestParams(size_t)
        {
            Storage<int> s;
            int *a, *b, *c;
            s.reserve(2);
            a = s.ptr();
            s.push_back(0);
            s.push_back(0);
            b = s.ptr();
            s.push_back(0);
            c = s.ptr();
            NNTestEquals(s.size(), 3);
            NNTestEquals(a, b);
            NNTestNotEquals(b, c);
        }
    }

    NNTestMethod(push_back)
    {
        NNTestParams(const T &)
        {
            Storage<int> s;
            NNTestEquals(s.push_back(42), s);
            NNTestEquals(s.size(), 1);
            NNTestEquals(s[0], 42);
        }
    }

    NNTestMethod(pop_back)
    {
        NNTestParams(const T &)
        {
            Storage<int> s(5, 42);
            NNTestEquals(s, s.pop_back());
            NNTestEquals(s.size(), 4);
        }
    }

    NNTestMethod(append)
    {
        NNTestParams(const Storage &)
        {
            Storage<int> s(2, 42);
            Storage<int> t(3, -3);
            NNTestEquals(s.append(t), s);
            NNTestEquals(s.size(), 5);
            NNTestEquals(s[0], 42);
            NNTestEquals(s[1], 42);
            NNTestEquals(s[2], -3);
            NNTestEquals(s[3], -3);
            NNTestEquals(s[4], -3);
        }
    }

    NNTestMethod(erase)
    {
        NNTestParams(size_t)
        {
            Storage<int> s({ 0, 1, 2, 3, 4, 5 });
            NNTestEquals(s.erase(0), s);
            NNTestEquals(s.size(), 5);
            NNTestEquals(s[0], 1);
            s.erase(1);
            NNTestEquals(s.size(), 4);
            NNTestEquals(s[1], 3);
            s.erase(3);
            NNTestEquals(s.size(), 3);
            NNTestEquals(s[2], 4);
        }
    }
}

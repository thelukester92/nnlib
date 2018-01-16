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
}

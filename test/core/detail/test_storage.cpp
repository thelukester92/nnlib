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
            NNTestEquals(s.size(), 0ul);
        }

        NNTestParams(size_t)
        {
            Storage<int> s(1);
            NNTestEquals(s.size(), 1ul);
            NNTestEquals(s[0], 0);
        }

        NNTestParams(size_t, T)
        {
            Storage<int> s(5, 42);
            NNTestEquals(s.size(), 5ul);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(s[i], 42);
        }

        NNTestParams(const Storage &)
        {
            Storage<int> s(5, 42);
            Storage<int> t(s);
            NNTestEquals(t.size(), 5ul);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(t[i], 42);
        }

        NNTestParams(Storage &&)
        {
            Storage<int> s(5, 42);
            Storage<int> t(std::move(s));
            NNTestEquals(t.size(), 5ul);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(t[i], 42);
        }

        NNTestParams(const std::initializer_list &)
        {
            Storage<int> s({ 0, 1, 2, 3, 4, 5 });
            NNTestEquals(s.size(), 6ul);
            for(int i = 0; i < 6; ++i)
                NNTestEquals(s[i], i);
        }

        NNTestParams(const Serialized &)
        {
            Storage<int> s(5, 42);
            Storage<int> t((Serialized(s)));
            NNTestEquals(t.size(), 5ul);
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
            NNTestEquals(t.size(), 5ul);
            for(size_t i = 0; i < 5; ++i)
                NNTestEquals(t[i], 42);
        }

        NNTestParams(const std::initializer_list<T> &)
        {
            Storage<int> s;
            s = { 0, 1, 2, 3, 4, 5 };
            NNTestEquals(s.size(), 6ul);
            for(int i = 0; i < 6; ++i)
                NNTestEquals(s[i], i);
        }
    }

    NNTestMethod(resize)
    {
        NNTestParams(size_t, const T &)
        {
            Storage<int> s;
            s.resize(5, 42);
            NNTestEquals(s.size(), 5ul);
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
            s.push(0);
            s.push(0);
            b = s.ptr();
            s.push(0);
            c = s.ptr();
            NNTestEquals(s.size(), 3ul);
            NNTestEquals(a, b);
            NNTestNotEquals(b, c);
        }
    }

    NNTestMethod(push)
    {
        NNTestParams(const T &)
        {
            Storage<int> s;
            NNTestEquals(s.push(42), s);
            NNTestEquals(s.size(), 1ul);
            NNTestEquals(s[0], 42);
        }
    }

    NNTestMethod(pop)
    {
        NNTestParams(const T &)
        {
            Storage<int> s(5, 42);
            NNTestEquals(s, s.pop());
            NNTestEquals(s.size(), 4ul);
        }
    }

    NNTestMethod(append)
    {
        NNTestParams(const Storage &)
        {
            Storage<int> s(2, 42);
            Storage<int> t(3, -3);
            NNTestEquals(s.append(t), s);
            NNTestEquals(s.size(), 5ul);
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
            NNTestEquals(s.size(), 5ul);
            NNTestEquals(s[0], 1);
            s.erase(1);
            NNTestEquals(s.size(), 4ul);
            NNTestEquals(s[1], 3);
            s.erase(3);
            NNTestEquals(s.size(), 3ul);
            NNTestEquals(s[2], 4);
        }
    }

    NNTestMethod(clear)
    {
        NNTestParams()
        {
            Storage<int> s(5, 42);
            NNTestEquals(s.clear(), s);
            NNTestEquals(s.size(), 0ul);
        }
    }

    NNTestMethod(ptr)
    {
        NNTestParams()
        {
            Storage<int> s, t;
            const Storage<int> &u = s;
            NNTestNotEquals(s.ptr(), t.ptr());
            NNTestEquals(s.ptr(), u.ptr());
        }
    }

    NNTestMethod(size)
    {
        NNTestParams()
        {
            Storage<int> s, t(1), u(2, 3);
            NNTestEquals(s.size(), 0ul);
            NNTestEquals(t.size(), 1ul);
            NNTestEquals(u.size(), 2ul);
        }
    }

    NNTestMethod(operator==)
    {
        NNTestParams(const Storage &)
        {
            Storage<int> s = { 1, 2 }, t = { 1, 2 }, u = { 1, 3 }, v;
            NNTest(s == t);
            NNTest(!(s == u));
            NNTest(!(s == v));
        }
    }

    NNTestMethod(operator!=)
    {
        NNTestParams(const Storage &)
        {
            Storage<int> s = { 1, 2 }, t = { 1, 2 }, u = { 1, 3 }, v;
            NNTest(!(s != t));
            NNTest(s != u);
            NNTest(s != v);
        }
    }

    NNTestMethod(at)
    {
        NNTestParams(size_t)
        {
            Storage<int> s = { 0, 1, 2, 3, 4, 5 };
            const Storage<int> &t = s;
            for(int i = 0; i < 6; ++i)
            {
                NNTestEquals(s.at(i), i);
                NNTestEquals(t.at(i), i);
                s.at(i) = 42;
                NNTestEquals(s.at(i), 42);
                NNTestEquals(t.at(i), 42);
            }
        }
    }

    NNTestMethod(operator[])
    {
        NNTestParams(size_t)
        {
            Storage<int> s = { 0, 1, 2, 3, 4, 5 };
            const Storage<int> &t = s;
            for(int i = 0; i < 6; ++i)
            {
                NNTestEquals(s[i], i);
                NNTestEquals(t[i], i);
                s[i] = 42;
                NNTestEquals(s[i], 42);
                NNTestEquals(t[i], 42);
            }
        }
    }

    NNTestMethod(front)
    {
        NNTestParams()
        {
            Storage<int> s = { 0, 1, 2, 3, 4, 5 };
            const Storage<int> &t = s;
            NNTestEquals(s.front(), 0);
            NNTestEquals(t.front(), 0);
            s.front() = 42;
            NNTestEquals(s.front(), 42);
            NNTestEquals(t.front(), 42);
        }
    }

    NNTestMethod(back)
    {
        NNTestParams()
        {
            Storage<int> s = { 0, 1, 2, 3, 4, 5 };
            const Storage<int> &t = s;
            NNTestEquals(s.back(), 5);
            NNTestEquals(t.back(), 5);
            s.back() = 42;
            NNTestEquals(s.back(), 42);
            NNTestEquals(t.back(), 42);
        }
    }

    NNTestMethod(begin)
    {
        NNTestParams()
        {
            Storage<int> s = { 0, 1, 2, 3, 4, 5 };
            const Storage<int> &t = s;
            NNTestEquals(s.begin(), s.ptr());
            NNTestEquals(t.begin(), s.ptr());
        }
    }

    NNTestMethod(end)
    {
        NNTestParams()
        {
            Storage<int> s = { 0, 1, 2, 3, 4, 5 };
            const Storage<int> &t = s;
            NNTestEquals(s.end(), s.ptr() + 6);
            NNTestEquals(t.end(), s.ptr() + 6);
        }
    }

    NNTestMethod(save)
    {
        NNTestParams(Serialized &)
        {
            Serialized s;
            Storage<int>({ 0, 1, 2, 3, 4, 5 }).save(s);
            Storage<int> t(s);
            NNTestEquals(t.size(), 6ul);
            for(int i = 0; i < 6; ++i)
                NNTestEquals(t[i], i);
        }
    }
}

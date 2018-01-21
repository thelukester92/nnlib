#include "../test_tensor.hpp"
#include "nnlib/core/tensor.hpp"
using namespace nnlib;
using T = NN_REAL_T;

NNTestClassImpl(Tensor)
{
    NNTestMethod(vectorize)
    {
        NNTestParams(const Storage<Tensor *> &)
        {
            Tensor<T> a = { 0, 1, 2, 3 }, b = { 4, 5 };
            Tensor<T> c = Tensor<T>({ 6, 7 }).resize(2, 1);
            Tensor<T> d = Tensor<T>::vectorize(Storage<Tensor<T> *>({ &a, &b, &c }));
            NNTestEquals(d.size(), 8);
            NNTestEquals(d.dims(), 1);
            NNTestEquals(d.sharedCount(), 4);
            NNTest(d.sharedWith(a));
            NNTest(d.sharedWith(b));
            NNTest(d.sharedWith(c));
            for(size_t i = 0; i < 8; ++i)
                NNTestEquals(d(i), i);
            Tensor<T> e = Tensor<T>::vectorize(Storage<Tensor<T> *>({ &a, &b, &c }));
            NNTestEquals(d.ptr(), e.ptr());
        }
    }

    NNTestMethod(concatenate)
    {
        NNTestParams(const Storage<Tensor *> &)
        {
            Tensor<T> a = Tensor<T>({ 0, 1, 2, 5, 6, 7 }).resize(2, 3);
            Tensor<T> b = Tensor<T>({ 3, 4, 8, 9 }).resize(2, 2);
            Tensor<T> c = Tensor<T>::concatenate(Storage<Tensor<T> *>({ &a, &b }));
            NNTestEquals(c.size(), 10);
            NNTestEquals(c.size(0), 2);
            NNTestEquals(c.size(1), 5);
            NNTestEquals(c.sharedCount(), 3);
            NNTest(c.sharedWith(a));
            NNTest(c.sharedWith(b));
            for(size_t i = 0; i < 2; ++i)
                for(size_t j = 0; j < 5; ++j)
                    NNTestEquals(c(i, j), (5 * i + j) % 10);
            Tensor<T> d = Tensor<T>::concatenate(Storage<Tensor<T> *>());
            NNTestEquals(d.size(), 0);
        }

        NNTestParams(const Storage<Tensor *> &, size_t)
        {
            Tensor<T> a = Tensor<T>({ 0, 1, 2, 3, 4, 5 }).resize(3, 2);
            Tensor<T> b = Tensor<T>({ 6, 7, 8, 9 }).resize(2, 2);
            Tensor<T> c = Tensor<T>::concatenate(Storage<Tensor<T> *>({ &a, &b }), 0);
            NNTestEquals(c.size(), 10);
            NNTestEquals(c.size(0), 5);
            NNTestEquals(c.size(1), 2);
            NNTestEquals(c.sharedCount(), 3);
            NNTest(c.sharedWith(a));
            NNTest(c.sharedWith(b));
            for(size_t i = 0; i < 5; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(c(i, j), (2 * i + j) % 10);
        }
    }

    NNTestMethod(Tensor)
    {
        NNTestParams()
        {
            Tensor<T> t;
            NNTestEquals(t.size(), 0);
            NNTestEquals(t.dims(), 1);
            NNTestEquals(t.size(0), 0);
        }

        NNTestParams(const Storage &)
        {
            Tensor<T> t(Storage<T>({ 0, 1, 2, 3, 4, 5 }));
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), i);
        }

        NNTestParams(const std::initializer_list &)
        {
            Tensor<T> t(Storage<T>({ 0, 1, 2, 3, 4, 5 }));
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), i);
        }

        NNTestParams(const Storage<size_t> &, bool)
        {
            Tensor<T> t(Storage<size_t>({ 3, 2 }), true);
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.size(0), 3);
            NNTestEquals(t.size(1), 2);
        }

        NNTestParams(size_t, size_t)
        {
            Tensor<T> t(3, 2);
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.size(0), 3);
            NNTestEquals(t.size(1), 2);
        }

        NNTestParams(Tensor &)
        {
            Tensor<T> t(Storage<T>({ 0, 1, 2, 3, 4, 5 }));
            Tensor<T> s(t);
            NNTest(t.sharedWith(s));
            NNTestEquals(s.size(), 6);
            NNTestEquals(s.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s(i), i);
        }

        NNTestParams(Tensor &&)
        {
            Tensor<T> t(Storage<T>({ 0, 1, 2, 3, 4, 5 }));
            Tensor<T> s(std::move(t));
            NNTest(t.sharedWith(s));
            NNTestEquals(s.size(), 6);
            NNTestEquals(s.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s(i), i);
        }

        NNTestParams(const Serialized &)
        {
            Tensor<T> t((Serialized(Tensor<T>(Storage<T>({ 0, 1, 2, 3, 4, 5 })))));
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), i);
            Serialized s;
            s.push(new Serialized());
            s.push(new Serialized());
            s.get(0)->push(0);
            s.get(0)->push(1);
            s.get(0)->push(2);
            s.get(1)->push(3);
            s.get(1)->push(4);
            s.get(1)->push(5);
            Tensor<T> u(s);
            NNTestEquals(u.dims(), 2);
            NNTestEquals(u.size(0), 2);
            NNTestEquals(u.size(1), 3);
            for(size_t i = 0; i < 2; ++i)
                for(size_t j = 0; j < 3; ++j)
                    NNTestEquals(u(i, j), 3 * i + j);
        }
    }

    NNTestMethod(operator=)
    {
        NNTestParams(const Storage &)
        {
            Tensor<T> t;
            t = Storage<T>({ 0, 1, 2, 3, 4, 5 });
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), i);
        }

        NNTestParams(const std::initializer_list &)
        {
            Tensor<T> t;
            t = { 0, 1, 2, 3, 4, 5 };
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), i);
        }

        NNTestParams(Tensor &)
        {
            Tensor<T> t(Storage<T>({ 0, 1, 2, 3, 4, 5 }));
            Tensor<T> s;
            s = t;
            NNTest(t.sharedWith(s));
            NNTestEquals(s.size(), 6);
            NNTestEquals(s.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s(i), i);
        }

        NNTestParams(Tensor &&)
        {
            Tensor<T> t(Storage<T>({ 0, 1, 2, 3, 4, 5 }));
            Tensor<T> s;
            s = std::move(t);
            NNTest(t.sharedWith(s));
            NNTestEquals(s.size(), 6);
            NNTestEquals(s.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s(i), i);
        }
    }

    NNTestMethod(shared)
    {
        NNTestParams()
        {
            Tensor<T> t, s;
            NNTest(!t.shared());
            NNTest(!s.shared());
            t = s;
            NNTest(t.shared());
            NNTest(s.shared());
        }
    }

    NNTestMethod(sharedWith)
    {
        NNTestParams(const Tensor &)
        {
            Tensor<T> t, s;
            NNTest(!t.sharedWith(s));
            NNTest(!s.sharedWith(t));
            NNTestNotEquals(t.ptr(), s.ptr());
            t = s;
            NNTest(t.sharedWith(s));
            NNTest(s.sharedWith(t));
            NNTestEquals(t.ptr(), s.ptr());
        }

        NNTestParams(const Storage<Tensor *> &)
        {
            Tensor<T> t, s, r;
            NNTest(!t.sharedWith({ &s, &r }));
            NNTestNotEquals(t.ptr(), s.ptr());
            NNTestNotEquals(t.ptr(), r.ptr());
            s = t;
            NNTest(!t.sharedWith({ &s, &r }));
            NNTestEquals(t.ptr(), s.ptr());
            r = s;
            NNTest(t.sharedWith({ &s, &r }));
            NNTestEquals(t.ptr(), s.ptr());
            NNTestEquals(t.ptr(), r.ptr());
        }
    }

    NNTestMethod(sharedCount)
    {
        NNTestParams()
        {
            Tensor<T> t, s, r;
            NNTestEquals(t.sharedCount(), 1);
            s = t;
            NNTestEquals(t.sharedCount(), 2);
            r = s;
            NNTestEquals(t.sharedCount(), 3);
        }
    }

    NNTestMethod(resize)
    {
        NNTestParams(const Storage<size_t> &)
        {
            Tensor<T> t;
            t.resize(Storage<size_t>({ 3, 2 }));
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.size(0), 3);
            NNTestEquals(t.size(1), 2);
            Tensor<T> s = t;
            NNTest(t.sharedWith(s));
            t.resize(10);
            NNTest(!t.sharedWith(s));
        }

        NNTestParams(size_t, size_t)
        {
            Tensor<T> t;
            t.resize(3, 2);
            NNTestEquals(t.size(), 6);
            NNTestEquals(t.size(0), 3);
            NNTestEquals(t.size(1), 2);
        }
    }

    NNTestMethod(resizeDim)
    {
        NNTestParams(size_t, size_t)
        {
            Tensor<T> t;
            t.resizeDim(0, 5);
            NNTestEquals(t.size(), 5);
            NNTestEquals(t.dims(), 1);
            t.resizeDim(0, 5);
            NNTestEquals(t.size(), 5);
            NNTestEquals(t.dims(), 1);
        }
    }

    NNTestMethod(view)
    {
        NNTestParams(const Storage<size_t> &)
        {
            Tensor<T> t({ 7, 1, 2, 3, 4, 5 });
            Tensor<T> v = t.view(Storage<size_t>({ 3, 2 }));
            v(0, 0) = 0;
            NNTestEquals(t(0), 0);
            NNTestEquals(v.size(), 6);
            NNTestEquals(v.dims(), 2);
            NNTestEquals(v.size(0), 3);
            NNTestEquals(v.size(1), 2);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(v(i, j), 2 * i + j);
            const Tensor<T> &u = const_cast<const Tensor<T> &>(t).view(Storage<size_t>({ 3, 2 }));
            NNTestEquals(u.size(), 6);
            NNTestEquals(u.dims(), 2);
            NNTestEquals(u.size(0), 3);
            NNTestEquals(u.size(1), 2);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(u(i, j), 2 * i + j);
        }

        NNTestParams(size_t, size_t)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 });
            Tensor<T> v = t.view(3, 2);
            NNTestEquals(v.size(), 6);
            NNTestEquals(v.dims(), 2);
            NNTestEquals(v.size(0), 3);
            NNTestEquals(v.size(1), 2);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(v(i, j), 2 * i + j);
            const Tensor<T> &u = const_cast<const Tensor<T> &>(t).view(3, 2);
            NNTestEquals(u.size(), 6);
            NNTestEquals(u.dims(), 2);
            NNTestEquals(u.size(0), 3);
            NNTestEquals(u.size(1), 2);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(u(i, j), 2 * i + j);
        }
    }

    NNTestMethod(reshape)
    {
        NNTestParams(const Storage<size_t> &)
        {
            Tensor<T> t({ 7, 1, 2, 3, 4, 5 });
            Tensor<T> v = t.reshape(Storage<size_t>({ 3, 2 }));
            v(0, 0) = 0;
            NNTestEquals(t(0), 7);
            NNTestEquals(v.size(), 6);
            NNTestEquals(v.dims(), 2);
            NNTestEquals(v.size(0), 3);
            NNTestEquals(v.size(1), 2);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(v(i, j), 2 * i + j);
        }

        NNTestParams(size_t, size_t)
        {
            Tensor<T> t({ 7, 1, 2, 3, 4, 5 });
            Tensor<T> v = t.reshape(3, 2);
            v(0, 0) = 0;
            NNTestEquals(t(0), 7);
            NNTestEquals(v.size(), 6);
            NNTestEquals(v.dims(), 2);
            NNTestEquals(v.size(0), 3);
            NNTestEquals(v.size(1), 2);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(v(i, j), 2 * i + j);
        }
    }

    NNTestMethod(select)
    {
        NNTestParams(size_t, size_t)
        {
            Tensor<T> t = Tensor<T>({ 7, 1, 2, 3, 4, 5 }).resize(3, 2);
            Tensor<T> v = t.select(0, 0);
            v(0) = 0;
            NNTestEquals(t(0, 0), 0);
            NNTestEquals(v.size(), 2);
            NNTestEquals(v.dims(), 1);
            NNTestEquals(v(0), 0);
            NNTestEquals(v(1), 1);
            v = t.select(0, 1);
            NNTestEquals(v.size(), 2);
            NNTestEquals(v.dims(), 1);
            NNTestEquals(v(0), 2);
            NNTestEquals(v(1), 3);
            v = t.select(0, 2);
            NNTestEquals(v.size(), 2);
            NNTestEquals(v.dims(), 1);
            NNTestEquals(v(0), 4);
            NNTestEquals(v(1), 5);
            v = t.select(1, 0);
            NNTestEquals(v.size(), 3);
            NNTestEquals(v.dims(), 1);
            NNTestEquals(v(0), 0);
            NNTestEquals(v(1), 2);
            NNTestEquals(v(2), 4);
            v = t.select(1, 1);
            NNTestEquals(v.size(), 3);
            NNTestEquals(v.dims(), 1);
            NNTestEquals(v(0), 1);
            NNTestEquals(v(1), 3);
            NNTestEquals(v(2), 5);
            const Tensor<T> &u = const_cast<const Tensor<T> &>(t).select(0, 0);
            NNTestEquals(u.size(), 2);
            NNTestEquals(u.dims(), 1);
            NNTestEquals(u(0), 0);
            NNTestEquals(u(1), 1);
            const Tensor<T> &w = const_cast<const Tensor<T> &>(t).select(0, 1);
            NNTestEquals(w.size(), 2);
            NNTestEquals(w.dims(), 1);
            NNTestEquals(w(0), 2);
            NNTestEquals(w(1), 3);
            const Tensor<T> &x = const_cast<const Tensor<T> &>(t).select(0, 2);
            NNTestEquals(x.size(), 2);
            NNTestEquals(x.dims(), 1);
            NNTestEquals(x(0), 4);
            NNTestEquals(x(1), 5);
            const Tensor<T> &y = const_cast<const Tensor<T> &>(t).select(1, 0);;
            NNTestEquals(y.size(), 3);
            NNTestEquals(y.dims(), 1);
            NNTestEquals(y(0), 0);
            NNTestEquals(y(1), 2);
            NNTestEquals(y(2), 4);
            const Tensor<T> &z = const_cast<const Tensor<T> &>(t).select(1, 1);
            NNTestEquals(z.size(), 3);
            NNTestEquals(z.dims(), 1);
            NNTestEquals(z(0), 1);
            NNTestEquals(z(1), 3);
            NNTestEquals(z(2), 5);
        }
    }

    NNTestMethod(narrow)
    {
        NNTestParams(size_t, size_t, size_t)
        {
            Tensor<T> t = Tensor<T>({ 0, 1, 7, 3, 4, 5 }).resize(3, 2);
            Tensor<T> v = t.narrow(0, 1, 2);
            v(0, 0) = 2;
            NNTestEquals(t(1, 0), 2);
            NNTestEquals(v.size(), 4);
            NNTestEquals(v.dims(), 2);
            NNTestEquals(v(0, 0), 2);
            NNTestEquals(v(0, 1), 3);
            NNTestEquals(v(1, 0), 4);
            NNTestEquals(v(1, 1), 5);
            const Tensor<T> &u = const_cast<const Tensor<T> &>(t).narrow(0, 1, 2);
            NNTestEquals(u.size(), 4);
            NNTestEquals(u.dims(), 2);
            NNTestEquals(u(0, 0), 2);
            NNTestEquals(u(0, 1), 3);
            NNTestEquals(u(1, 0), 4);
            NNTestEquals(u(1, 1), 5);
        }
    }

    NNTestMethod(expand)
    {
        NNTestParams(size_t, size_t)
        {
            Tensor<T> t = Tensor<T>({ 7, 1, 2, 3, 4, 5 }).resize(1, 6);
            Tensor<T> v = t.expand(0, 3);
            v(0, 0) = 0;
            NNTestEquals(t(0, 0), 0);
            NNTestEquals(v.size(), 18);
            NNTestEquals(v.dims(), 2);
            NNTestEquals(v.size(0), 3);
            NNTestEquals(v.size(1), 6);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 6; ++j)
                    NNTestEquals(&v(i, j), &t(0, j));
            const Tensor<T> &u = const_cast<const Tensor<T> &>(t).expand(0, 3);
            NNTestEquals(u.size(), 18);
            NNTestEquals(u.dims(), 2);
            NNTestEquals(u.size(0), 3);
            NNTestEquals(u.size(1), 6);
            for(size_t i = 0; i < 3; ++i)
                for(size_t j = 0; j < 6; ++j)
                    NNTestEquals(&u(i, j), &t(0, j));
        }
    }

    NNTestMethod(sub)
    {
        NNTestParams(Tensor &, const std::initializer_list<const std::initializer_list<size_t>> &)
        {
            Tensor<T> t = Tensor<T>({ 0, 7, 2, 3, 4, 5 }).resize(2, 3), s;
            t.sub(s, { {}, { 1, 2 } });
            s(0, 0) = 0;
            NNTestEquals(t(0, 1), 0);
            NNTestEquals(s.size(), 4);
            NNTestEquals(s.dims(), 2);
            NNTestEquals(s.size(0), 2);
            NNTestEquals(s.size(1), 2);
            for(size_t i = 0; i < 2; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(&s(i, j), &t(i, j + 1));
            t.sub(s, { { 0 }, { 1, 2 } });
            NNTestEquals(s.size(), 2);
            NNTestEquals(s.dims(), 2);
            NNTestEquals(s.size(0), 1);
            NNTestEquals(s.size(1), 2);
            for(size_t i = 0; i < 1; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(&s(i, j), &t(i, j + 1));
        }

        NNTestParams(const std::initializer_list<const std::initializer_list<size_t>> &)
        {
            Tensor<T> t = Tensor<T>({ 0, 7, 2, 3, 4, 5 }).resize(2, 3);
            Tensor<T> s = t.sub({ {}, { 1, 2 } });
            s(0, 0) = 0;
            NNTestEquals(t(0, 1), 0);
            NNTestEquals(s.size(), 4);
            NNTestEquals(s.dims(), 2);
            NNTestEquals(s.size(0), 2);
            NNTestEquals(s.size(1), 2);
            for(size_t i = 0; i < 2; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(&s(i, j), &t(i, j + 1));
            const Tensor<T> &u = const_cast<const Tensor<T> &>(t).sub({ { 0 }, { 1, 2 } });
            NNTestEquals(u.size(), 2);
            NNTestEquals(u.dims(), 2);
            NNTestEquals(u.size(0), 1);
            NNTestEquals(u.size(1), 2);
            for(size_t i = 0; i < 1; ++i)
                for(size_t j = 0; j < 2; ++j)
                    NNTestEquals(&u(i, j), &t(i, j + 1));
        }
    }

    NNTestMethod(copy)
    {
        NNTestParams()
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 });
            Tensor<T> s = t.copy();
            NNTest(!t.sharedWith(s));
            NNTestEquals(s.size(), 6);
            NNTestEquals(s.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s(i), i);
        }

        NNTestParams(const Tensor &)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 }), s(6);
            s.copy(t);
            NNTest(!t.sharedWith(s));
            NNTestEquals(s.size(), 6);
            NNTestEquals(s.dims(), 1);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(s(i), i);
        }
    }

    NNTestMethod(swap)
    {
        NNTestParams(Tensor &)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 }), s(6);
            NNTestEquals(&s.swap(t), &s);
            for(size_t i = 0; i < 6; ++i)
            {
                NNTestEquals(t(i), 0);
                NNTestEquals(s(i), i);
            }
        }

        NNTestParams(Tensor &&)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 }), s(6);
            NNTestEquals(&s.swap(std::move(t)), &s);
            for(size_t i = 0; i < 6; ++i)
            {
                NNTestEquals(t(i), 0);
                NNTestEquals(s(i), i);
            }
        }
    }

    NNTestMethod(transpose)
    {
        NNTestParams(size_t, size_t)
        {
            Tensor<T> t = Tensor<T>({ 0, 1, 2, 3, 4, 5 }).resize(2, 3);
            Tensor<T> s = t.transpose(0, 1);
            for(size_t i = 0; i < 2; ++i)
            {
                for(size_t j = 0; j < 3; ++j)
                {
                    NNTestEquals(t(i, j), 3 * i + j);
                    NNTestEquals(s(j, i), 3 * i + j);
                    NNTestEquals(&t(i, j), &s(j, i));
                }
            }
            const Tensor<T> &u = const_cast<const Tensor<T> &>(t).transpose(0, 1);
            for(size_t i = 0; i < 2; ++i)
            {
                for(size_t j = 0; j < 3; ++j)
                {
                    NNTestEquals(u(j, i), 3 * i + j);
                    NNTestEquals(&t(i, j), &u(j, i));
                }
            }
        }
    }

    NNTestMethod(shape)
    {
        NNTestParams()
        {
            Tensor<T> t(3, 2);
            NNTestEquals(t.shape(), Storage<size_t>({ 3, 2 }));
        }
    }

    NNTestMethod(strides)
    {
        NNTestParams()
        {
            Tensor<T> t(3, 2);
            NNTestEquals(t.strides(), Storage<size_t>({ 2, 1 }));
            NNTestEquals(t.transpose().strides(), Storage<size_t>({ 1, 2 }));
        }
    }

    NNTestMethod(dims)
    {
        NNTestParams()
        {
            Tensor<T> t, u(1), v(1, 2), w(1, 2, 3);
            NNTestEquals(t.dims(), 1);
            NNTestEquals(u.dims(), 1);
            NNTestEquals(v.dims(), 2);
            NNTestEquals(w.dims(), 3);
        }
    }

    NNTestMethod(size)
    {
        NNTestParams()
        {
            Tensor<T> t, u(1), v(1, 2), w(1, 2, 3);
            NNTestEquals(t.size(), 0);
            NNTestEquals(u.size(), 1);
            NNTestEquals(v.size(), 2);
            NNTestEquals(w.size(), 6);
        }

        NNTestParams(size_t)
        {
            Tensor<T> t(1, 2, 3);
            NNTestEquals(t.size(0), 1);
            NNTestEquals(t.size(1), 2);
            NNTestEquals(t.size(2), 3);
        }
    }

    NNTestMethod(contiguous)
    {
        NNTestParams()
        {
            Tensor<T> t(3, 2);
            NNTest(t.contiguous());
            NNTest(!t.transpose().contiguous());
        }
    }

    NNTestMethod(makeContiguous)
    {
        NNTestParams()
        {
            Tensor<T> t(3, 2);
            T *ptr = t.ptr();
            t.makeContiguous();
            NNTestEquals(ptr, t.ptr());
            Tensor<T> s = t.transpose();
            ptr = s.ptr();
            s.makeContiguous();
            NNTestNotEquals(ptr, s.ptr());
            NNTest(s.contiguous());
        }
    }

    NNTestMethod(stride)
    {
        NNTestParams(size_t)
        {
            Tensor<T> t(3, 2);
            NNTestEquals(t.stride(0), 2);
            NNTestEquals(t.stride(1), 1);
        }
    }

    NNTestMethod(fill)
    {
        NNTestParams(const T &)
        {
            Tensor<T> t(6);
            NNTestEquals(&t.fill(0), &t);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), 0);
            t.fill(3.14);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), 3.14);
        }
    }

    NNTestMethod(zeros)
    {
        NNTestParams()
        {
            Tensor<T> t(6);
            NNTestEquals(&t.zeros(), &t);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), 0);
        }
    }

    NNTestMethod(ones)
    {
        NNTestParams()
        {
            Tensor<T> t(6);
            NNTestEquals(&t.ones(), &t);
            for(size_t i = 0; i < 6; ++i)
                NNTestEquals(t(i), 1);
        }
    }

    NNTestMethod(scale)
    {
        NNTestParams(T)
        {
            Tensor<T> t = Tensor<T>(3).ones();
            t.scale(3.14);
            for(size_t i = 0; i < t.size(); ++i)
                NNTestEquals(t(i), 3.14);
        }
    }

    NNTestMethod(add)
    {
        NNTestParams(T)
        {
            Tensor<T> t(3);
            t.push(3.14);
            for(size_t i = 0; i < t.size(); ++i)
                NNTestEquals(t(i), 3.14);
        }
    }

    NNTestMethod(at)
    {
        NNTestParams(const Storage<size_t> &)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 });
            const Tensor<T> &s = t;
            for(size_t i = 0; i < t.size(); ++i)
            {
                NNTestEquals(t.at(Storage<size_t>({ i })), i);
                NNTestEquals(s.at(Storage<size_t>({ i })), i);
            }
        }

        NNTestParams(size_t)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 });
            const Tensor<T> &s = t;
            for(size_t i = 0; i < t.size(); ++i)
            {
                NNTestEquals(t.at(i), i);
                NNTestEquals(s.at(i), i);
            }
        }
    }

    NNTestMethod(operator())
    {
        NNTestParams(const Storage<size_t> &)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 });
            const Tensor<T> &s = t;
            for(size_t i = 0; i < t.size(); ++i)
            {
                NNTestEquals(t(Storage<size_t>({ i })), i);
                NNTestEquals(s(Storage<size_t>({ i })), i);
            }
        }

        NNTestParams(size_t)
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 });
            const Tensor<T> &s = t;
            for(size_t i = 0; i < t.size(); ++i)
            {
                NNTestEquals(t(i), i);
                NNTestEquals(s(i), i);
            }
        }
    }

    NNTestMethod(ptr)
    {
        NNTestParams()
        {
            Tensor<T> t(1);
            const Tensor<T> &s = t;
            NNTestEquals(t.ptr(), s.ptr());
            NNTestEquals(t.ptr(), &t(0));
        }
    }

    NNTestMethod(data)
    {
        NNTestParams()
        {
            Tensor<T> t({ 3.14 });
            const Tensor<T> &s = t;
            NNTestEquals(t.data(), s.data());
            NNTestEquals(t.data()[0], 3.14);
            NNTestEquals(s.data()[0], 3.14);
        }
    }

    NNTestMethod(begin)
    {
        NNTestParams()
        {
            Tensor<T> t({ 3.14 });
            const Tensor<T> &s = t;
            NNTestEquals(*t.begin(), *s.begin());
            NNTestEquals(*t.begin(), 3.14);
            NNTestEquals(*s.begin(), 3.14);
        }
    }

    NNTestMethod(end)
    {
        NNTestParams()
        {
            Tensor<T> t({ 0, 1, 2, 3, 4, 5 });
            const Tensor<T> &s = t;
            NNTestEquals(std::distance(t.begin(), t.end()), (int) t.size());
            size_t idx = 0;
            for(auto i = t.begin(); i != t.end(); ++i, ++idx)
                NNTestEquals(*i, idx);
            idx = 0;
            for(auto i = s.begin(); i != s.end(); ++i, ++idx)
                NNTestEquals(*i, idx);
        }
    }

    NNTestMethod(save)
    {
        Serialized s;
        Tensor<T>(Storage<T>({ 0, 1, 2, 3, 4, 5 })).save(s);
        Tensor<T> t(s);
        NNTestEquals(t.size(), 6);
        NNTestEquals(t.dims(), 1);
        for(size_t i = 0; i < 6; ++i)
            NNTestEquals(t(i), i);
    }
}

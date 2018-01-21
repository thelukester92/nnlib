#include "../test_algebra.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/math/algebra.hpp"
#include "nnlib/math/math.hpp"
using namespace nnlib;
using namespace nnlib::math;
using T = NN_REAL_T;

NNTestClassImpl(Algebra)
{
    NNTestMethod(vFill)
    {
        NNTestParams(Tensor &x, T)
        {
            Tensor<T> t(5);
            vFill(t, 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
            t = Tensor<T>(5, 3).select(1, 1);
            vFill(t, 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }

        NNTestParams(Tensor &&x, T)
        {
            Tensor<T> t(5);
            vFill(std::move(t), 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }
    }

    NNTestMethod(vScale)
    {
        NNTestParams(Tensor &x, T)
        {
            Tensor<T> t({ 1, 1, 1, 1, 1, 1 });
            vScale(t, 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
            t = math::fill(Tensor<T>(5, 3).select(1, 1), 1);
            vScale(t, 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }

        NNTestParams(Tensor &&x, T)
        {
            Tensor<T> t({ 1, 1, 1, 1, 1, 1 });
            vScale(std::move(t), 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }
    }

    NNTestMethod(mFill)
    {
        NNTestParams(Tensor &x, T)
        {
            Tensor<T> t(5, 3);
            mFill(t, 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }

        NNTestParams(Tensor &&x, T)
        {
            Tensor<T> t(5, 3);
            mFill(std::move(t), 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }
    }

    NNTestMethod(mScale)
    {
        NNTestParams(Tensor &x, T)
        {
            Tensor<T> t = Tensor<T>({ 1, 1, 1, 1, 1, 1 }).resize(2, 3);
            mScale(t, 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }

        NNTestParams(Tensor &&x, T)
        {
            Tensor<T> t = Tensor<T>({ 1, 1, 1, 1, 1, 1 }).resize(2, 3);
            mScale(std::move(t), 3.14);
            forEach([&](T t)
            {
                NNTestAlmostEquals(t, 3.14, 1e-12);
            }, t);
        }
    }

    NNTestMethod(vAdd_v)
    {
        NNTestParams(cosnt Tensor &, Tensor &, T)
        {
            Tensor<T> t({ 1, 2, 3 });
            Tensor<T> u({ 4, 5, 6 });
            vAdd_v(u, t, 0.5);
            NNTestAlmostEquals(t(0), 3, 1e-12);
            NNTestAlmostEquals(t(1), 4.5, 1e-12);
            NNTestAlmostEquals(t(2), 6, 1e-12);
            t = math::fill(Tensor<T>(3, 5).select(1, 1), 0);
            vAdd_v(u, t, 1);
            NNTestAlmostEquals(t(0), 4, 1e-12);
            NNTestAlmostEquals(t(1), 5, 1e-12);
            NNTestAlmostEquals(t(2), 6, 1e-12);
        }

        NNTestParams(cosnt Tensor &, Tensor &&, T)
        {
            Tensor<T> t({ 1, 2, 3 });
            Tensor<T> u({ 4, 5, 6 });
            vAdd_v(u, std::move(t), 0.5);
            NNTestAlmostEquals(t(0), 3, 1e-12);
            NNTestAlmostEquals(t(1), 4.5, 1e-12);
            NNTestAlmostEquals(t(2), 6, 1e-12);
        }

        NNTestParams(cosnt Tensor &, Tensor &, T, T)
        {
            Tensor<T> t({ 1, 2, 4 });
            Tensor<T> u({ 4, 5, 6 });
            vAdd_v(u, t, 0.5, 0.25);
            NNTestAlmostEquals(t(0), 2.25, 1e-12);
            NNTestAlmostEquals(t(1), 3, 1e-12);
            NNTestAlmostEquals(t(2), 4, 1e-12);
        }

        NNTestParams(cosnt Tensor &, Tensor &&, T, T)
        {
            Tensor<T> t({ 1, 2, 4 });
            Tensor<T> u({ 4, 5, 6 });
            vAdd_v(u, std::move(t), 0.5, 0.25);
            NNTestAlmostEquals(t(0), 2.25, 1e-12);
            NNTestAlmostEquals(t(1), 3, 1e-12);
            NNTestAlmostEquals(t(2), 4, 1e-12);
        }
    }

    NNTestMethod(mAdd_vv)
    {
        NNTestParams(const Tensor &, const Tensor &, Tensor &, T, T)
        {
            Tensor<T> t({ 1, 2, 3 });
            Tensor<T> u({ 4, 5 });
            Tensor<T> A = Tensor<T>({ 1, 1, 1, 1, 1, 1 }).resize(3, 2);
            mAdd_vv(t, u, A, 0.5, 0.25);
            for(size_t i = 0; i < A.size(0); ++i)
                for(size_t j = 0; j < A.size(1); ++j)
                    NNTestAlmostEquals(A(i, j), 0.5 * t(i) * u(j) + 0.25, 1e-12);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> t({ 1, 2, 3 });
            Tensor<T> u({ 4, 5 });
            Tensor<T> A = Tensor<T>({ 1, 1, 1, 1, 1, 1 }).resize(3, 2);
            mAdd_vv(t, u, std::move(A), 0.5, 0.25);
            for(size_t i = 0; i < A.size(0); ++i)
                for(size_t j = 0; j < A.size(1); ++j)
                    NNTestAlmostEquals(A(i, j), 0.5 * t(i) * u(j) + 0.25, 1e-12);
        }
    }

    NNTestMethod(vAdd_mv)
    {
        NNTestParams(const Tensor &, const Tensor &, Tensor &, T, T)
        {
            Tensor<T> t({ 1, 2, 3 }), u({ 1, 1 });
            Tensor<T> A = Tensor<T>({ 4, 5, 6, 7, 8, 9 }).resize(2, 3);
            vAdd_mv(A, t, u, 0.5, 0.25);
            NNTestAlmostEquals(u(0), 16.25, 1e-12);
            NNTestAlmostEquals(u(1), 25.25, 1e-12);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> t({ 1, 2, 3 }), u({ 1, 1 });
            Tensor<T> A = Tensor<T>({ 4, 5, 6, 7, 8, 9 }).resize(2, 3);
            vAdd_mv(A, t, std::move(u), 0.5, 0.25);
            NNTestAlmostEquals(u(0), 16.25, 1e-12);
            NNTestAlmostEquals(u(1), 25.25, 1e-12);
        }
    }

    NNTestMethod(vAdd_mtv)
    {
        NNTestParams(const Tensor &, const Tensor &, Tensor &, T, T)
        {
            Tensor<T> t({ 1, 2, 3 }), u({ 1, 1 });
            Tensor<T> A = Tensor<T>({ 4, 7, 5, 8, 6, 9 }).resize(3, 2);
            vAdd_mtv(A, t, u, 0.5, 0.25);
            NNTestAlmostEquals(u(0), 16.25, 1e-12);
            NNTestAlmostEquals(u(1), 25.25, 1e-12);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> t({ 1, 2, 3 }), u({ 1, 1 });
            Tensor<T> A = Tensor<T>({ 4, 7, 5, 8, 6, 9 }).resize(3, 2);
            vAdd_mtv(A, t, std::move(u), 0.5, 0.25);
            NNTestAlmostEquals(u(0), 16.25, 1e-12);
            NNTestAlmostEquals(u(1), 25.25, 1e-12);
        }
    }

    NNTestMethod(mAdd_m)
    {
        NNTestParams(const Tensor &, Tensor &, T, T)
        {
            Tensor<T> A = Tensor<T>({ 2, 4, 6, 8, 10, 12 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 1, 2, 4, 8, 16, 32 }).resize(2, 3);
            mAdd_m(B, A, 0.5, 0.25);
            NNTestAlmostEquals(A(0, 0), 1, 1e-12);
            NNTestAlmostEquals(A(0, 1), 2, 1e-12);
            NNTestAlmostEquals(A(0, 2), 3.5, 1e-12);
            NNTestAlmostEquals(A(1, 0), 6, 1e-12);
            NNTestAlmostEquals(A(1, 1), 10.5, 1e-12);
            NNTestAlmostEquals(A(1, 2), 19, 1e-12);
        }

        NNTestParams(const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> A = Tensor<T>({ 2, 4, 6, 8, 10, 12 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 1, 2, 4, 8, 16, 32 }).resize(2, 3);
            mAdd_m(B, std::move(A), 0.5, 0.25);
            NNTestAlmostEquals(A(0, 0), 1, 1e-12);
            NNTestAlmostEquals(A(0, 1), 2, 1e-12);
            NNTestAlmostEquals(A(0, 2), 3.5, 1e-12);
            NNTestAlmostEquals(A(1, 0), 6, 1e-12);
            NNTestAlmostEquals(A(1, 1), 10.5, 1e-12);
            NNTestAlmostEquals(A(1, 2), 19, 1e-12);
        }
    }

    NNTestMethod(mAdd_mt)
    {
        NNTestParams(const Tensor &, Tensor &, T, T)
        {
            Tensor<T> A = Tensor<T>({ 2, 4, 6, 8, 10, 12 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 1, 8, 2, 16, 4, 32 }).resize(3, 2);
            mAdd_mt(B, A, 0.5, 0.25);
            NNTestAlmostEquals(A(0, 0), 1, 1e-12);
            NNTestAlmostEquals(A(0, 1), 2, 1e-12);
            NNTestAlmostEquals(A(0, 2), 3.5, 1e-12);
            NNTestAlmostEquals(A(1, 0), 6, 1e-12);
            NNTestAlmostEquals(A(1, 1), 10.5, 1e-12);
            NNTestAlmostEquals(A(1, 2), 19, 1e-12);
        }

        NNTestParams(const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> A = Tensor<T>({ 2, 4, 6, 8, 10, 12 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 1, 8, 2, 16, 4, 32 }).resize(3, 2);
            mAdd_mt(B, std::move(A), 0.5, 0.25);
            NNTestAlmostEquals(A(0, 0), 1, 1e-12);
            NNTestAlmostEquals(A(0, 1), 2, 1e-12);
            NNTestAlmostEquals(A(0, 2), 3.5, 1e-12);
            NNTestAlmostEquals(A(1, 0), 6, 1e-12);
            NNTestAlmostEquals(A(1, 1), 10.5, 1e-12);
            NNTestAlmostEquals(A(1, 2), 19, 1e-12);
        }
    }

    NNTestMethod(mAdd_mm)
    {
        NNTestParams(const Tensor &, const Tensor &, Tensor &, T, T)
        {
            Tensor<T> A = Tensor<T>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 7, 8, 9, 0, 1, 2 }).resize(3, 2);
            Tensor<T> C = Tensor<T>({ 2, 4, 6, 8 }).resize(2, 2);
            Tensor<T> D = Tensor<T>({ 29, 16, 82, 48 }).resize(2, 2);
            mAdd_mm(A, B, C, 1, 0.5);
            forEach([&](T c, T d)
            {
                NNTestAlmostEquals(c, d, 1e-12);
            }, C, D);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> A = Tensor<T>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 7, 8, 9, 0, 1, 2 }).resize(3, 2);
            Tensor<T> C = Tensor<T>({ 2, 4, 6, 8 }).resize(2, 2);
            Tensor<T> D = Tensor<T>({ 29, 16, 82, 48 }).resize(2, 2);
            mAdd_mm(A, B, std::move(C), 1, 0.5);
            forEach([&](T c, T d)
            {
                NNTestAlmostEquals(c, d, 1e-12);
            }, C, D);
        }
    }

    NNTestMethod(mAdd_mtm)
    {
        NNTestParams(const Tensor &, const Tensor &, Tensor &, T, T)
        {
            Tensor<T> A = Tensor<T>({ 1, 4, 2, 5, 3, 6 }).resize(3, 2);
            Tensor<T> B = Tensor<T>({ 7, 8, 9, 0, 1, 2 }).resize(3, 2);
            Tensor<T> C = Tensor<T>({ 2, 4, 6, 8 }).resize(2, 2);
            Tensor<T> D = Tensor<T>({ 29, 16, 82, 48 }).resize(2, 2);
            mAdd_mtm(A, B, C, 1, 0.5);
            forEach([&](T c, T d)
            {
                NNTestAlmostEquals(c, d, 1e-12);
            }, C, D);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> A = Tensor<T>({ 1, 4, 2, 5, 3, 6 }).resize(3, 2);
            Tensor<T> B = Tensor<T>({ 7, 8, 9, 0, 1, 2 }).resize(3, 2);
            Tensor<T> C = Tensor<T>({ 2, 4, 6, 8 }).resize(2, 2);
            Tensor<T> D = Tensor<T>({ 29, 16, 82, 48 }).resize(2, 2);
            mAdd_mtm(A, B, std::move(C), 1, 0.5);
            forEach([&](T c, T d)
            {
                NNTestAlmostEquals(c, d, 1e-12);
            }, C, D);
        }
    }

    NNTestMethod(mAdd_mmt)
    {
        NNTestParams(const Tensor &, const Tensor &, Tensor &, T, T)
        {
            Tensor<T> A = Tensor<T>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 7, 9, 1, 8, 0, 2 }).resize(2, 3);
            Tensor<T> C = Tensor<T>({ 2, 4, 6, 8 }).resize(2, 2);
            Tensor<T> D = Tensor<T>({ 29, 16, 82, 48 }).resize(2, 2);
            mAdd_mmt(A, B, C, 1, 0.5);
            forEach([&](T c, T d)
            {
                NNTestAlmostEquals(c, d, 1e-12);
            }, C, D);
        }

        NNTestParams(const Tensor &, const Tensor &, Tensor &&, T, T)
        {
            Tensor<T> A = Tensor<T>({ 1, 2, 3, 4, 5, 6 }).resize(2, 3);
            Tensor<T> B = Tensor<T>({ 7, 9, 1, 8, 0, 2 }).resize(2, 3);
            Tensor<T> C = Tensor<T>({ 2, 4, 6, 8 }).resize(2, 2);
            Tensor<T> D = Tensor<T>({ 29, 16, 82, 48 }).resize(2, 2);
            mAdd_mmt(A, B, std::move(C), 1, 0.5);
            forEach([&](T c, T d)
            {
                NNTestAlmostEquals(c, d, 1e-12);
            }, C, D);
        }
    }
}

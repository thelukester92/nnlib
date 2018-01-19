#ifndef MATH_ALGEBRA_TPP
#define MATH_ALGEBRA_TPP

#include "../algebra.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/math/random.hpp"

namespace nnlib
{

namespace math
{

template <typename T>
void vFill(Tensor<T> &_x, typename traits::Identity<T>::type alpha)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    T *x = _x.ptr();
    size_t n = _x.size(), s = _x.stride(0);
    for(size_t i = 0; i < n; ++i)
        x[i * s] = alpha;
}

template <typename T>
void vFill(Tensor<T> &&_x, typename traits::Identity<T>::type alpha)
{
    vFill(_x, alpha);
}

template <typename T>
void vScale(Tensor<T> &&_x, typename traits::Identity<T>::type alpha)
{
    vScale(_x, alpha);
}

template <typename T>
void mFill(Tensor<T> &_A, typename traits::Identity<T>::type alpha)
{
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    T *A = _A.ptr();
    size_t r = _A.size(0), c = _A.size(1), ld = _A.stride(0);
    for(size_t i = 0; i < r; ++i)
        for(size_t j = 0; j < c; ++j)
            A[i * ld + j] = alpha;
}

template <typename T>
void mFill(Tensor<T> &&_A, typename traits::Identity<T>::type alpha)
{
    mFill(_A, alpha);
}

template <typename T>
void mScale(Tensor<T> &&_A, typename traits::Identity<T>::type alpha)
{
    mScale(_A, alpha);
}

template <typename T>
void vAdd_v(const Tensor<T> &_x, Tensor<T> &&_y, typename traits::Identity<T>::type alpha)
{
    vAdd_v(_x, _y, alpha);
}

template <typename T>
void vAdd_v(const Tensor<T> &_x, Tensor<T> &&_y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    vAdd_v(_x, _y, alpha, beta);
}

template <typename T>
void mAdd_vv(const Tensor<T> &_x, const Tensor<T> &_y, Tensor<T> &&_A, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    mAdd_vv(_x, _y, _A, alpha, beta);
}

template <typename T>
void vAdd_mv(const Tensor<T> &_A, const Tensor<T> &_x, Tensor<T> &&_y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    vAdd_mv(_A, _x, _y, alpha, beta);
}

template <typename T>
void vAdd_mtv(const Tensor<T> &_A, const Tensor<T> &_x, Tensor<T> &&_y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    vAdd_mtv(_A, _x, _y, alpha, beta);
}

template <typename T>
void mAdd_m(const Tensor<T> &_A, Tensor<T> &_B, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
    NNAssertEquals(_A.shape(), _B.shape(), "Incompatible operands!");
    const T *A = _A.ptr();
    T *B = _B.ptr();
    size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), ldb = _B.stride(0);
    for(size_t i = 0; i < r; ++i)
        for(size_t j = 0; j < c; ++j)
            B[i * ldb + j] = alpha * A[i * lda + j] + beta * B[i * ldb + j];
}

template <typename T>
void mAdd_m(const Tensor<T> &_A, Tensor<T> &&_B, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    mAdd_m(_A, _B, alpha, beta);
}

template <typename T>
void mAdd_mt(const Tensor<T> &_A, Tensor<T> &_B, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
    NNAssertEquals(_A.size(0), _B.size(1), "Incompatible operands!");
    NNAssertEquals(_A.size(1), _B.size(0), "Incompatible operands!");
    const T *A = _A.ptr();
    T *B = _B.ptr();
    size_t r = _A.size(1), c = _A.size(0), lda = _A.stride(0), ldb = _B.stride(0);
    for(size_t i = 0; i < r; ++i)
        for(size_t j = 0; j < c; ++j)
            B[i * ldb + j] = alpha * A[j * lda + i] + beta * B[i * ldb + j];
}

template <typename T>
void mAdd_mt(const Tensor<T> &_A, Tensor<T> &&_B, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    mAdd_mt(_A, _B, alpha, beta);
}

template <typename T>
void mAdd_mm(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &&_C, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    mAdd_mm(_A, _B, _C, alpha, beta);
}

template <typename T>
void mAdd_mtm(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &&_C, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    mAdd_mtm(_A, _B, _C, alpha, beta);
}

template <typename T>
void mAdd_mmt(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &&_C, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    mAdd_mmt(_A, _B, _C, alpha, beta);
}

#ifndef NN_ACCEL_CPU
    template <typename T>
    void vScale(Tensor<T> &_x, typename traits::Identity<T>::type alpha)
    {
        NNAssertEquals(_x.dims(), 1, "Expected a vector!");
        T *x = _x.ptr();
        size_t n = _x.size(), s = _x.stride(0);
        for(size_t i = 0; i < n; ++i)
            x[i * s] *= alpha;
    }

    template <typename T>
    void mScale(Tensor<T> &_A, typename traits::Identity<T>::type alpha)
    {
        NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
        T *A = _A.ptr();
        size_t r = _A.size(0), c = _A.size(1), ld = _A.stride(0);
        for(size_t i = 0; i < r; ++i)
            for(size_t j = 0; j < c; ++j)
                A[i * ld + j] *= alpha;
    }

    template <typename T>
    void vAdd_v(const Tensor<T> &_x, Tensor<T> &_y, typename traits::Identity<T>::type alpha)
    {
        NNAssertEquals(_x.dims(), 1, "Expected a vector!");
        NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
        const T *x = _x.ptr();
        T *y = _y.ptr();
        size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
        for(size_t i = 0; i < n; ++i)
            y[i * sy] = alpha * x[i * sx] + y[i * sy];
    }

    template <typename T>
    void vAdd_v(const Tensor<T> &_x, Tensor<T> &_y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
    {
        NNAssertEquals(_x.dims(), 1, "Expected a vector!");
        NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
        const T *x = _x.ptr();
        T *y = _y.ptr();
        size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
        for(size_t i = 0; i < n; ++i)
            y[i * sy] = alpha * x[i * sx] + beta * y[i * sy];
    }

    template <typename T>
    void mAdd_vv(const Tensor<T> &_x, const Tensor<T> &_y, Tensor<T> &_A, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
    {
        NNAssertEquals(_x.dims(), 1, "Expected a vector!");
        NNAssertEquals(_y.dims(), 1, "Expected a vector!");
        NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
        NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
        const T *x = _x.ptr();
        const T *y = _y.ptr();
        T *A = _A.ptr();
        size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
        for(size_t i = 0; i < r; ++i)
            for(size_t j = 0; j < c; ++j)
                A[i * lda + j] = alpha * x[i * sx] * y[j * sy] + beta * A[i * lda + j];
    }

    template <typename T>
    void vAdd_mv(const Tensor<T> &_A, const Tensor<T> &_x, Tensor<T> &_y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
    {
        NNAssertEquals(_x.dims(), 1, "Expected a vector!");
        NNAssertEquals(_y.dims(), 1, "Expected a vector!");
        NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_x.size(), _A.size(1), "Incompatible operands!");
        NNAssertEquals(_y.size(), _A.size(0), "Incompatible operands!");
        const T *A = _A.ptr();
        const T *x = _x.ptr();
        T *y = _y.ptr();
        size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
        for(size_t i = 0; i < r; ++i)
        {
            T &v = y[i * sy];
            v *= beta;
            for(size_t j = 0; j < c; ++j)
                v += alpha * A[i * lda + j] * x[j * sx];
        }
    }

    template <typename T>
    void vAdd_mtv(const Tensor<T> &_A, const Tensor<T> &_x, Tensor<T> &_y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
    {
        NNAssertEquals(_x.dims(), 1, "Expected a vector!");
        NNAssertEquals(_y.dims(), 1, "Expected a vector!");
        NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
        NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
        const T *A = _A.ptr();
        const T *x = _x.ptr();
        T *y = _y.ptr();
        size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
        for(size_t i = 0; i < c; ++i)
        {
            T &v = y[i * sy];
            v *= beta;
            for(size_t j = 0; j < r; ++j)
                v += alpha * A[j * lda + i] * x[j * sx];
        }
    }

    template <typename T>
    void mAdd_mm(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &_C, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
    {
        NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_A.size(0), _C.size(0), "Incompatible operands!");
        NNAssertEquals(_A.size(1), _B.size(0), "Incompatible operands!");
        NNAssertEquals(_B.size(1), _C.size(1), "Incompatible operands!");
        const T *A = _A.ptr();
        const T *B = _B.ptr();
        T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        for(size_t i = 0; i < M; ++i)
        {
            for(size_t j = 0; j < N; ++j)
            {
                T &v = C[i * ldc + j];
                v *= beta;
                for(size_t k = 0; k < K; ++k)
                {
                    v += alpha * A[i * lda + k] * B[k * ldb + j];
                }
            }
        }
    }

    template <typename T>
    void mAdd_mtm(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &_C, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
    {
        NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_A.size(1), _C.size(0), "Incompatible operands!");
        NNAssertEquals(_A.size(0), _B.size(0), "Incompatible operands!");
        NNAssertEquals(_B.size(1), _C.size(1), "Incompatible operands!");
        const T *A = _A.ptr();
        const T *B = _B.ptr();
        T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(0);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        for(size_t i = 0; i < M; ++i)
        {
            for(size_t j = 0; j < N; ++j)
            {
                T &v = C[i * ldc + j];
                v *= beta;
                for(size_t k = 0; k < K; ++k)
                {
                    v += alpha * A[k * lda + i] * B[k * ldb + j];
                }
            }
        }
    }

    template <typename T>
    void mAdd_mmt(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &_C, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
    {
        NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_B.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_B.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_C.dims(), 2, "Expected a matrix!");
        NNAssertEquals(_C.stride(1), 1, "Expected a contiguous leading dimension!");
        NNAssertEquals(_A.size(0), _C.size(0), "Incompatible operands!");
        NNAssertEquals(_A.size(1), _B.size(1), "Incompatible operands!");
        NNAssertEquals(_B.size(0), _C.size(1), "Incompatible operands!");
        const T *A = _A.ptr();
        const T *B = _B.ptr();
        T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        for(size_t i = 0; i < M; ++i)
        {
            for(size_t j = 0; j < N; ++j)
            {
                T &v = C[i * ldc + j];
                v *= beta;
                for(size_t k = 0; k < K; ++k)
                {
                    v += alpha * A[i * lda + k] * B[j * ldb + k];
                }
            }
        }
    }
#endif

} // namespace math

} // namespace nnlib

#endif

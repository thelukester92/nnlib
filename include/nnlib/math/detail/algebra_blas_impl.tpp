// WARNING: This file assumes it has been included by algebra_blas_(float|double)_impl.tpp

namespace nnlib
{

namespace math
{

template <typename T>
void vScale(Tensor<T> &_x, typename traits::Identity<T>::type alpha)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    T *x = _x.ptr();
    size_t n = _x.size(), s = _x.stride(0);
    scal(n, alpha, x, s);
}

template <typename T>
void mScale(Tensor<T> &_A, typename traits::Identity<T>::type alpha)
{
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    T *A = _A.ptr();
    size_t r = _A.size(0), c = _A.size(1), ld = _A.stride(0);
    for(size_t i = 0; i < r; ++i, A += ld)
        scal(c, alpha, A, 1);
}

template <typename T>
void vAdd_v(const Tensor<T> &_x, Tensor<T> &_y, typename traits::Identity<T>::type alpha)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
    const T *x = _x.ptr();
    T *y = _y.ptr();
    size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
    axpy(n, alpha, x, sx, y, sy);
}

template <typename T>
void vAdd_v(const Tensor<T> &_x, Tensor<T> &_y, typename traits::Identity<T>::type alpha, typename traits::Identity<T>::type beta)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
    const T *x = _x.ptr();
    T *y = _y.ptr();
    size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
    axpby(n, alpha, x, sx, beta, y, sy);
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
    if(beta != 1)
        mScale(_A, beta);
    ger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
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
    gemv(CblasRowMajor, CblasNoTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
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
    gemv(CblasRowMajor, CblasTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
}

#ifndef NN_ACCEL_GPU
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
        T *A = const_cast<T *>(_A.ptr());
        T *B = const_cast<T *>(_B.ptr());
        T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
        T *A = const_cast<T *>(_A.ptr());
        T *B = const_cast<T *>(_B.ptr());
        T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(0);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        gemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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
        T *A = const_cast<T *>(_A.ptr());
        T *B = const_cast<T *>(_B.ptr());
        T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
#endif

} // namespace math

} // namespace nnlib

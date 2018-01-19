// WARNING: This file assumes it has been included by algebra_blas_(float|double)_impl.tpp

namespace nnlib
{

template <>
void Algebra<NN_REAL_T>::vScale(Tensor<NN_REAL_T> &_x, NN_REAL_T alpha)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NN_REAL_T *x = _x.ptr();
    size_t n = _x.size(), s = _x.stride(0);
    scal(n, alpha, x, s);
}

template <>
void Algebra<NN_REAL_T>::mScale(Tensor<NN_REAL_T> &_A, NN_REAL_T alpha)
{
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    NN_REAL_T *A = _A.ptr();
    size_t r = _A.size(0), c = _A.size(1), ld = _A.stride(0);
    for(size_t i = 0; i < r; ++i, A += ld)
        scal(c, alpha, A, 1);
}

template <>
void Algebra<NN_REAL_T>::vAdd_v(const Tensor<NN_REAL_T> &_x, Tensor<NN_REAL_T> &_y, NN_REAL_T alpha)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
    const NN_REAL_T *x = _x.ptr();
    NN_REAL_T *y = _y.ptr();
    size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
    axpy(n, alpha, x, sx, y, sy);
}

template <>
void Algebra<NN_REAL_T>::vAdd_v(const Tensor<NN_REAL_T> &_x, Tensor<NN_REAL_T> &_y, NN_REAL_T alpha, NN_REAL_T beta)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NNAssertEquals(_x.shape(), _y.shape(), "Incompatible operands!");
    const NN_REAL_T *x = _x.ptr();
    NN_REAL_T *y = _y.ptr();
    size_t n = _x.size(), sx = _x.stride(0), sy = _y.stride(0);
    axpby(n, alpha, x, sx, beta, y, sy);
}

template <>
void Algebra<NN_REAL_T>::mAdd_vv(const Tensor<NN_REAL_T> &_x, const Tensor<NN_REAL_T> &_y, Tensor<NN_REAL_T> &_A, NN_REAL_T alpha, NN_REAL_T beta)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NNAssertEquals(_y.dims(), 1, "Expected a vector!");
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
    NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
    const NN_REAL_T *x = _x.ptr();
    const NN_REAL_T *y = _y.ptr();
    NN_REAL_T *A = _A.ptr();
    size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
    if(beta != 1)
        mScale(_A, beta);
    ger(CblasRowMajor, r, c, alpha, x, sx, y, sy, A, lda);
}

template <>
void Algebra<NN_REAL_T>::vAdd_mv(const Tensor<NN_REAL_T> &_A, const Tensor<NN_REAL_T> &_x, Tensor<NN_REAL_T> &_y, NN_REAL_T alpha, NN_REAL_T beta)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NNAssertEquals(_y.dims(), 1, "Expected a vector!");
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    NNAssertEquals(_x.size(), _A.size(1), "Incompatible operands!");
    NNAssertEquals(_y.size(), _A.size(0), "Incompatible operands!");
    const NN_REAL_T *A = _A.ptr();
    const NN_REAL_T *x = _x.ptr();
    NN_REAL_T *y = _y.ptr();
    size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
    gemv(CblasRowMajor, CblasNoTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
}

template <>
void Algebra<NN_REAL_T>::vAdd_mtv(const Tensor<NN_REAL_T> &_A, const Tensor<NN_REAL_T> &_x, Tensor<NN_REAL_T> &_y, NN_REAL_T alpha, NN_REAL_T beta)
{
    NNAssertEquals(_x.dims(), 1, "Expected a vector!");
    NNAssertEquals(_y.dims(), 1, "Expected a vector!");
    NNAssertEquals(_A.dims(), 2, "Expected a matrix!");
    NNAssertEquals(_A.stride(1), 1, "Expected a contiguous leading dimension!");
    NNAssertEquals(_x.size(), _A.size(0), "Incompatible operands!");
    NNAssertEquals(_y.size(), _A.size(1), "Incompatible operands!");
    const NN_REAL_T *A = _A.ptr();
    const NN_REAL_T *x = _x.ptr();
    NN_REAL_T *y = _y.ptr();
    size_t r = _A.size(0), c = _A.size(1), lda = _A.stride(0), sx = _x.stride(0), sy = _y.stride(0);
    gemv(CblasRowMajor, CblasTrans, r, c, alpha, A, lda, x, sx, beta, y, sy);
}

#ifndef NN_ACCEL_GPU
    template <>
    void Algebra<NN_REAL_T>::mAdd_mm(const Tensor<NN_REAL_T> &_A, const Tensor<NN_REAL_T> &_B, Tensor<NN_REAL_T> &_C, NN_REAL_T alpha, NN_REAL_T beta)
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
        NN_REAL_T *A = const_cast<NN_REAL_T *>(_A.ptr());
        NN_REAL_T *B = const_cast<NN_REAL_T *>(_B.ptr());
        NN_REAL_T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template <>
    void Algebra<NN_REAL_T>::mAdd_mtm(const Tensor<NN_REAL_T> &_A, const Tensor<NN_REAL_T> &_B, Tensor<NN_REAL_T> &_C, NN_REAL_T alpha, NN_REAL_T beta)
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
        NN_REAL_T *A = const_cast<NN_REAL_T *>(_A.ptr());
        NN_REAL_T *B = const_cast<NN_REAL_T *>(_B.ptr());
        NN_REAL_T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(0);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        gemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template <>
    void Algebra<NN_REAL_T>::mAdd_mmt(const Tensor<NN_REAL_T> &_A, const Tensor<NN_REAL_T> &_B, Tensor<NN_REAL_T> &_C, NN_REAL_T alpha, NN_REAL_T beta)
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
        NN_REAL_T *A = const_cast<NN_REAL_T *>(_A.ptr());
        NN_REAL_T *B = const_cast<NN_REAL_T *>(_B.ptr());
        NN_REAL_T *C = _C.ptr();
        size_t M = _C.size(0), N = _C.size(1), K = _A.size(1);
        size_t lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
        gemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
#endif

}

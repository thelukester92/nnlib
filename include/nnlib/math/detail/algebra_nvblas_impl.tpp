// WARNING: This file assumes it has been included by algebra_nvblas_(T|double)_impl.tpp

namespace nnlib
{

namespace math
{

template <typename T>
void mAdd_mm(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &_C, T alpha, T beta)
{
    const T *A = _A.ptr();
    const T *B = _B.ptr();
    T *C = _C.ptr();
    int M = _C.size(0), N = _C.size(1), K = _A.size(1), lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
    gemm("N", "N", &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

template <typename T>
void mAdd_mtm(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &_C, T alpha, T beta)
{
    const T *A = _A.ptr();
    const T *B = _B.ptr();
    T *C = _C.ptr();
    int M = _C.size(0), N = _C.size(1), K = _A.size(0), lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
    gemm("N", "T", &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}
template <typename T>
void mAdd_mmt(const Tensor<T> &_A, const Tensor<T> &_B, Tensor<T> &_C, T alpha, T beta)
{
    const T *A = _A.ptr();
    const T *B = _B.ptr();
    T *C = _C.ptr();
    int M = _C.size(0), N = _C.size(1), K = _A.size(1), lda = _A.stride(0), ldb = _B.stride(0), ldc = _C.stride(0);
    gemm("T", "N", &N, &M, &K, &alpha, B, &ldb, A, &lda, &beta, C, &ldc);
}

} // namespace math

} // namespace nnlib

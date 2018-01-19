// WARNING: This file assumes it has been included by algebra_nvblas_(float|double)_impl.tpp

namespace nnlib
{

namespace math
{

template <typename T>
void mAdd_mm(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    gemm("N", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

template <typename T>
void mAdd_mtm(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    gemm("N", "T", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}
template <typename T>
void mAdd_mmt(const Tensor<float> &_A, const Tensor<float> &_B, Tensor<float> &_C, float alpha, float beta)
{
    int m = (int) M, n = (int) N, k = (int) K, a = (int) lda, b = (int) ldb, c = (int) ldc;
    gemm("T", "N", &n, &m, &k, &alpha, B, &b, A, &a, &beta, C, &c);
}

} // namespace math

} // namespace nnlib

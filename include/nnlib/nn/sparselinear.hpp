#ifndef NN_SPARSE_LINEAR_HPP
#define NN_SPARSE_LINEAR_HPP

#include "linear.hpp"

namespace nnlib
{

/// \brief A feed-forward layer that returns a linear combination of sparse inputs.
///
/// See nnlib/util/tensor_util.hpp for an explanation of sparse tensors.
/// Like a Linear layer, a SparseLinear layer expects either vector or matrix input,
/// so the input must be a matrix that encodes a vector or matrix.
template <typename T = NN_REAL_T>
class SparseLinear : public Linear<T>
{
public:
	using Linear<T>::Linear;
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override;
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	
	using Linear<T>::m_weights;
	using Linear<T>::m_weightsGrad;
	
	using Linear<T>::m_useBias;
	using Linear<T>::m_bias;
	using Linear<T>::m_biasGrad;
	
	using Linear<T>::m_ones;
};

}

NNRegisterType(SparseLinear, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::SparseLinear<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/sparselinear.tpp"
#endif

#endif

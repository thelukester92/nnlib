#ifndef NN_SOFT_MAX_HPP
#define NN_SOFT_MAX_HPP

#include "module.hpp"

namespace nnlib
{

/// \brief Softmax for classification problems.
///
/// When using NLL, LogSoftMax is preferred.
template <typename T = double>
class SoftMax : public Module<T>
{
public:
	SoftMax();
	SoftMax(const SoftMax &module);
	SoftMax(const Serialized &node);
	SoftMax &operator=(const SoftMax &module);
	
	virtual void save(Serialized &node) const override;
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override;
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(SoftMax, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::SoftMax<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/softmax.tpp"
#endif

#endif

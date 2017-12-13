#ifndef NN_LOG_SOFT_MAX_HPP
#define NN_LOG_SOFT_MAX_HPP

#include "module.hpp"

namespace nnlib
{

/// \brief Log softmax for classification problems.
///
/// Best combined with NLL critic. Works only on 2D tensors.
template <typename T = double>
class LogSoftMax : public Module<T>
{
public:
	LogSoftMax();
	LogSoftMax(const LogSoftMax &module);
	LogSoftMax(const Serialized &node);
	LogSoftMax &operator=(const LogSoftMax &module);
	
	virtual void save(Serialized &node) const override;
	virtual Tensor<T> &forward(const Tensor<T> &input) override;
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(LogSoftMax, Module);
NNTemplateDefinition(LogSoftMax, "detail/logsoftmax.tpp");

#endif

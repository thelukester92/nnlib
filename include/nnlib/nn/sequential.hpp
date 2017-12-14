#ifndef NN_SEQUENTIAL_HPP
#define NN_SEQUENTIAL_HPP

#include "container.hpp"

namespace nnlib
{

/// A standard feed-forward neural network module.
template <typename T = double>
class Sequential : public Container<T>
{
public:
	using Container<T>::Container;
	using Container<T>::components;
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override;
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
	virtual Tensor<T> &output() override;
	virtual Tensor<T> &inGrad() override;
	
protected:
	using Container<T>::m_components;
};

}

NNRegisterType(Sequential, Module);

#ifdef NN_REAL_T
	extern template class nnlib::Sequential<NN_REAL_T>;
#else
	#include "detail/sequential.tpp"
#endif

#endif

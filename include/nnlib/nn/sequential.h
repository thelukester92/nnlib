#ifndef NN_SEQUENTIAL_H
#define NN_SEQUENTIAL_H

#include "container.h"

namespace nnlib
{

/// A standard feed-forward neural network module.
template <typename T = double>
class Sequential : public Container<T>
{
public:
	using Container<T>::Container;
	using Container<T>::components;
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		Tensor<T> *inp = const_cast<Tensor<T> *>(&input);
		for(Module<T> *component : m_components)
			inp = &component->forward(*inp);
		return m_output = *inp;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		const Tensor<T> *grad = &outGrad;
		for(size_t i = components() - 1; i > 0; --i)
			grad = &m_components[i]->backward(m_components[i - 1]->output(), *grad);
		return m_inGrad = m_components[0]->backward(input, *grad);
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	using Container<T>::m_components;
};

}

NNRegisterType(Sequential, Module);

#endif

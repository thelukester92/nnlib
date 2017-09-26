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
	
	// MARK: Container methods
	
	/// Add a component to this container, enforcing compatibility.
	virtual Sequential &add(Module<T> *component) override
	{
		m_components.push_back(component);
		return *this;
	}
	
	/// Remove and return a specific component from this container, enforcing compatibility.
	virtual Module<T> *remove(size_t index) override
	{
		Module<T> *comp = m_components[index];
		m_components.erase(index);
		return comp;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual void updateOutput(const Tensor<T> &input) override
	{
		Tensor<T> *inp = const_cast<Tensor<T> *>(&input);
		for(Module<T> *component : m_components)
			inp = &component->forward(*inp);
		m_output = *inp;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual void updateGrad(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		const Tensor<T> *grad = &outGrad;
		for(size_t i = components() - 1; i > 0; --i)
			grad = &m_components[i]->backward(m_components[i - 1]->output(), *grad);
		m_inGrad = m_components[0]->backward(input, *grad);
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	using Container<T>::m_components;
};

}

NNRegisterType(Sequential, Module);

#endif

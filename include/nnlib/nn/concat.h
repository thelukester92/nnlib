#ifndef NN_CONCAT_H
#define NN_CONCAT_H

#include "container.h"

namespace nnlib
{

/// A container that concatenates the outputs of each component.
template <typename T = double>
class Concat : public Container<T>
{
using Container<T>::components;
using Container<T>::m_components;
public:
	using Container<T>::inputs;
	using Container<T>::outputs;
	using Container<T>::batch;
	
	Concat() {}
	
	template <typename ... Ms>
	Concat(Module<T> *component, Ms *...components)
	{
		add(component, components...);
	}
	
	/// Add multiple components.
	template <typename ... Ms>
	Concat &add(Module<T> *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	// MARK: Container methods
	
	/// Add a component to this container, enforcing compatibility.
	virtual Concat &add(Module<T> *component) override
	{
		NNAssert(components() == 0 || m_components[0]->inputs() == component->inputs(), "Incompatible concat component!");
		m_components.push_back(component);
		return resizeBuffers();
	}
	
	/// Remove and return a specific component from this container, enforcing compatibility.
	virtual Module<T> *remove(size_t index) override
	{
		Module<T> *comp = m_components[index];
		m_components.erase(index);
		resizeBuffers();
		return comp;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		for(Module<T> *component : m_components)
		{
			component->forward(input);
		}
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		m_inGrad.fill(0);
		
		size_t offset = 0, size;
		for(Module<T> *component : m_components)
		{
			size = component->outputs()[1];
			m_inGrad.addMM(component->backward(input, outGrad.sub({ {}, { offset, size } })));
			offset += size;
		}
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	virtual Concat &inputs(const Storage<size_t> &dims) override
	{
		for(Module<T> *component : m_components)
		{
			component->inputs(dims);
		}
		return resizeBuffers();
	}
	
	/// Set the output shape of this module, including batch.
	virtual Concat &outputs(const Storage<size_t> &dims) override
	{
		throw std::runtime_error("Cannot directly change concat outputs! Add or remove components instead.");
	}
	
private:
	Concat &resizeBuffers()
	{
		NNAssert(components() > 0 && m_components[0]->outputs().size() == 2, "Expected matrix IO for concat!");
		
		size_t outs = 0, bats = m_components[0]->outputs()[0], inps = m_components[0]->inputs()[1];
		for(Module<T> *component : m_components)
		{
			outs += component->outputs()[1];
		}
		
		m_output.resize(bats, outs);
		m_inGrad.resize(bats, inps);
		
		size_t offset = 0, size;
		for(Module<T> *component : m_components)
		{
			size = component->outputs()[1];
			component->output() = m_output.sub({ {}, { offset, size } });
			offset += size;
		}
		
		return *this;
	}
	
	Tensor<T> m_output;
	Tensor<T> m_inGrad;
};

}

#endif

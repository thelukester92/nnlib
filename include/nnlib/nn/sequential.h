#ifndef NN_SEQUENTIAL_H
#define NN_SEQUENTIAL_H

#include "container.h"

namespace nnlib
{

/// A standard feed-forward neural network module.
class Sequential : public Container
{
using Container::components;
using Container::m_components;
public:
	using Container::inputs;
	using Container::outputs;
	using Container::batch;
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "sequential";
	}
	
	Sequential() {}
	
	template <typename ... Ms>
	Sequential(Module *component, Ms *...components)
	{
		add(component, components...);
	}
	
	/// Add multiple components.
	template <typename ... Ms>
	Sequential &add(Module *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	// MARK: Container methods
	
	/// Add a component to this container, enforcing compatibility.
	virtual Sequential &add(Module *component) override
	{
		m_components.push_back(component);
		if(components() > 1)
		{
			resizeDown(m_components.size() - 1);
		}
		return *this;
	}
	
	/// Remove and return a specific component from this container, enforcing compatibility.
	virtual Module *remove(size_t index) override
	{
		Module *comp = m_components[index];
		m_components.erase(index);
		
		if(index > 0)
		{
			resizeUp(index - 1);
			resizeDown(index);
		}
		
		return comp;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) override
	{
		Tensor *inp = const_cast<Tensor *>(&input);
		for(Module *component : m_components)
		{
			inp = &component->forward(*inp);
		}
		return *inp;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) override
	{
		const Tensor *grad = &outGrad;
		for(size_t i = components() - 1; i > 0; --i)
		{
			grad = &m_components[i]->backward(m_components[i - 1]->output(), *grad);
		}
		return m_components[0]->backward(input, *grad);
	}
	
	/// Cached output.
	virtual Tensor &output() override
	{
		return m_components.back()->output();
	}
	
	/// Cached input gradient.
	virtual Tensor &inGrad() override
	{
		return m_components.front()->inGrad();
	}
	
	/// Set the input shape of this module, including batch.
	virtual Sequential &inputs(const Storage<size_t> &dims) override
	{
		m_components.front()->inputs(dims);
		resizeDown();
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	virtual Sequential &outputs(const Storage<size_t> &dims) override
	{
		m_components.back()->outputs(dims);
		resizeUp();
		return *this;
	}
	
private:
	Sequential &resizeDown(size_t start = 1)
	{
		for(size_t i = start, count = components(); i < count; ++i)
		{
			m_components[i]->inputs(m_components[i-1]->outputs());
		}
		return *this;
	}
	
	Sequential &resizeUp(size_t start = (size_t) -1)
	{
		start = std::min(start, components() - 1);
		for(size_t i = start; i > 0; --i)
		{
			m_components[i - 1]->outputs(m_components[i]->inputs());
		}
		return *this;
	}
};

}

#endif

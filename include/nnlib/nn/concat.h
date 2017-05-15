#ifndef NN_CONCAT_H
#define NN_CONCAT_H

#include "container.h"

namespace nnlib
{

/// A container that concatenates the outputs of each component.
class Concat : public Container
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
		return "concat";
	}
	
	Concat() {}
	
	template <typename ... Ms>
	Concat(Module *component, Ms *...components)
	{
		add(component, components...);
	}
	
	/// Add multiple components.
	template <typename ... Ms>
	Concat &add(Module *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	// MARK: Container methods
	
	/// Add a component to this container, enforcing compatibility.
	virtual Concat &add(Module *component) override
	{
		NNAssert(components() == 0 || m_components[0]->inputs() == component->inputs(), "Incompatible concat component!");
		m_components.push_back(component);
		return resizeBuffers();
	}
	
	/// Remove and return a specific component from this container, enforcing compatibility.
	virtual Module *remove(size_t index) override
	{
		Module *comp = m_components[index];
		m_components.erase(index);
		resizeBuffers();
		return comp;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor &forward(const Tensor &input) override
	{
		for(Module *component : m_components)
		{
			component->forward(input);
		}
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor &backward(const Tensor &input, const Tensor &outGrad) override
	{
		m_inGrad.fill(0);
		
		size_t offset = 0, size;
		for(Module *component : m_components)
		{
			size = component->outputs()[1];
			m_inGrad.addMM(component->backward(input, outGrad.sub({ {}, { offset, size } })));
			offset += size;
		}
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	virtual Concat &inputs(const Storage<size_t> &dims) override
	{
		for(Module *component : m_components)
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
		for(Module *component : m_components)
		{
			outs += component->outputs()[1];
		}
		
		m_output.resize(bats, outs);
		m_inGrad.resize(bats, inps);
		
		size_t offset = 0, size;
		for(Module *component : m_components)
		{
			size = component->outputs()[1];
			component->output() = m_output.sub({ {}, { offset, size } });
			offset += size;
		}
		
		return *this;
	}
	
	Tensor m_output;
	Tensor m_inGrad;
};

}

#endif

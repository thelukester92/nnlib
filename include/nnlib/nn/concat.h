#ifndef NN_CONCAT_H
#define NN_CONCAT_H

#include "container.h"

namespace nnlib
{

/// A container that concatenates the outputs of each component.
template <typename T = double>
class Concat : public Container<T>
{
using Container<T>::m_components;
public:
	using Container<T>::components;
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
	
	Concat(const Concat &module) :
		Container<T>(module)
	{
		resizeBuffers();
	}
	
	Concat &operator=(const Concat &module)
	{
		*static_cast<Container<T> *>(this) = module;
		resizeBuffers();
		return *this;
	}
	
	// MARK: Container methods
	
	/// Add a component to this container, enforcing compatibility.
	virtual Concat &add(Module<T> *component) override
	{
		NNAssert(components() == 0 || m_components[0]->inputs() == component->inputs(), "Incompatible component!");
		NNAssertEquals(component->inputs().size(), 2, "Expected matrix input!");
		NNAssertEquals(component->outputs().size(), 2, "Expected matrix output!");
		m_components.push_back(component);
		return resizeBuffers();
	}
	
	/// Remove and return a specific component from this container, enforcing compatibility.
	virtual Module<T> *remove(size_t index) override
	{
		Module<T> *comp = Container<T>::remove(index);
		if(components() > 0)
			resizeBuffers();
		return comp;
	}
	
	// MARK: Module methods
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		for(Module<T> *component : m_components)
			component->forward(input);
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
			m_inGrad.addM(component->backward(input, outGrad.sub({ {}, { offset, size } })));
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
		NNAssertEquals(dims.size(), 2, "Expected matrix input!");
		for(Module<T> *component : m_components)
			component->inputs(dims);
		return resizeBuffers();
	}
	
	/// Set the output shape of this module, including batch.
	virtual Concat &outputs(const Storage<size_t> &dims) override
	{
		throw Error("Cannot directly change concat outputs! Add or remove components instead.");
	}
	
	/// Set the batch size of this module.
	virtual Concat &batch(size_t bats) override
	{
		Container<T>::batch(bats);
		m_output.resizeDim(0, bats);
		m_inGrad.resizeDim(0, bats);
		return *this;
	}
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const override
	{
		node.set("components", m_components);
	}
	
	/// Load from a serialized node.
	virtual void load(const Serialized &node) override
	{
		this->clear();
		for(Module<T> *component : node.get<Storage<Module<T> *>>("components"))
			add(component);
	}
	
private:
	Concat &resizeBuffers()
	{
		NNAssertGreaterThan(components(), 0, "Expected at least one component!");
		
		size_t outs = 0, bats = m_components[0]->outputs()[0], inps = m_components[0]->inputs()[1];
		for(Module<T> *component : m_components)
			outs += component->outputs()[1];
		
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

NNRegisterType(Concat, Module);

#endif

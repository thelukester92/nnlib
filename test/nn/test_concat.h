#ifndef TEST_CONCAT_H
#define TEST_CONCAT_H

#include "nnlib/nn/concat.h"
#include "nnlib/nn/linear.h"
#include "nnlib/nn/identity.h"
using namespace nnlib;

void TestConcat()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({
		-5, 10,
		15, -20
	}).resize(2, 2);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({
		1, 2, 3, 4, -5,
		-4, -3, 2, 1, 7
	}).resize(2, 5);
	
	// Linear layer with weights and bias, arbitrary
	Linear<> *linear = new Linear<>(2, 3, 2);
	linear->weights().copy({
		-3, -2, 2,
		3, 4, 5
	});
	linear->bias().copy({ -5, 7, 8862.37 });
	
	// Identity layer
	Identity<> *identity = new Identity<>(2);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({
		45, 50, 40, -5, 10,
		-105, -110, -70, 15, -20
	}).resize(2, 5);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({
		???
	}).resize(2, 2);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		???
	});
	
	/// \todo test forward and backward using the parameters and targets above
	/// \todo test the stuff below
	
	/*
	
	/// Get a specific component from this container.
	Module<T> *component(size_t index)
	{
		return m_components[index];
	}
	
	/// Get the number of components in this container.
	size_t components() const
	{
		return m_components.size();
	}
	
	/// Add multiple components to this container.
	template <typename ... Ms>
	Container &add(Module<T> *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	/// Add a component to this container.
	virtual Container &add(Module<T> *component)
	{
		m_components.push_back(component);
		return *this;
	}
	
	/// Remove and return a specific component from this container.
	virtual Module<T> *remove(size_t index)
	{
		Module<T> *comp = m_components[index];
		m_components.erase(index);
		return comp;
	}
	
	/// Remove all components from this container.
	virtual Container &clear()
	{
		for(Module<T> *comp : m_components)
		{
			delete comp;
		}
		m_components.clear();
		return *this;
	}
	
	/// Set the batch size of this module.
	virtual Container &batch(size_t bats) override
	{
		for(Module<T> *component : m_components)
		{
			component->batch(bats);
		}
		return *this;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters.
	virtual Storage<Tensor<T> *> parameterList() override
	{
		Storage<Tensor<T> *> params;
		for(Module<T> *comp : m_components)
		{
			for(Tensor<T> *param : comp->parameterList())
			{
				params.push_back(param);
			}
		}
		return params;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's parameters' gradient.
	virtual Storage<Tensor<T> *> gradList() override
	{
		Storage<Tensor<T> *> blams;
		for(Module<T> *comp : m_components)
		{
			for(Tensor<T> *blam : comp->gradList())
			{
				blams.push_back(blam);
			}
		}
		return blams;
	}
	
	/// A vector of tensors filled with (views of) each sub-module's internal state.
	virtual Storage<Tensor<T> *> stateList() override
	{
		Storage<Tensor<T> *> states;
		for(Module<T> *comp : m_components)
		{
			for(Tensor<T> *state : comp->stateList())
			{
				states.push_back(state);
			}
		}
		return states;
	}
	
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
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<Concat>::name << m_components.size();
		for(Module<T> *component : m_components)
			out << *component;
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(
			str == Binding<Concat>::name,
			"Unexpected type! Expected '" + Binding<Concat>::name + "', got '" + str + "'!"
		);
		
		size_t n;
		in >> n;
		
		m_components.resize(n);
		for(size_t i = 0; i < n; ++i)
			in >> m_components[i];
	}
	
	*/
}

#endif

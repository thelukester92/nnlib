#ifndef NN_SEQUENTIAL_H
#define NN_SEQUENTIAL_H

#include "container.h"

namespace nnlib
{

/// A standard feed-forward neural network module.
template <typename T = double>
class Sequential : public Container<T>
{
using Container<T>::components;
using Container<T>::m_components;
public:
	using Container<T>::inputs;
	using Container<T>::outputs;
	using Container<T>::batch;
	
	Sequential() {}
	
	template <typename ... Ms>
	Sequential(Module<T> *component, Ms *...components)
	{
		add(component, components...);
	}
	
	/// Add multiple components.
	template <typename ... Ms>
	Sequential &add(Module<T> *component, Ms *...more)
	{
		add(component);
		add(more...);
		return *this;
	}
	
	// MARK: Container methods
	
	/// Add a component to this container, enforcing compatibility.
	virtual Sequential &add(Module<T> *component) override
	{
		m_components.push_back(component);
		if(components() > 1)
		{
			resizeDown(m_components.size() - 1);
		}
		return *this;
	}
	
	/// Remove and return a specific component from this container, enforcing compatibility.
	virtual Module<T> *remove(size_t index) override
	{
		Module<T> *comp = m_components[index];
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
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		Tensor<T> *inp = const_cast<Tensor<T> *>(&input);
		for(Module<T> *component : m_components)
		{
			inp = &component->forward(*inp);
		}
		return *inp;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		const Tensor<T> *grad = &outGrad;
		for(size_t i = components() - 1; i > 0; --i)
		{
			grad = &m_components[i]->backward(m_components[i - 1]->output(), *grad);
		}
		return m_components[0]->backward(input, *grad);
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_components.back()->output();
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
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
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<Sequential>::name << m_components.size();
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
			str == Binding<Sequential>::name,
			"Unexpected type! Expected '" + Binding<Sequential>::name + "', got '" + str + "'!"
		);
		
		size_t n;
		in >> n;
		
		m_components.resize(n);
		for(size_t i = 0; i < n; ++i)
			in >> m_components[i];
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

NNSerializable(Sequential<double>, Module<double>);
NNSerializable(Sequential<float>, Module<float>);

}

#endif

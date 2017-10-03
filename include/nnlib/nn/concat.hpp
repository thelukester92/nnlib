#ifndef NN_CONCAT_HPP
#define NN_CONCAT_HPP

#include "container.hpp"

namespace nnlib
{

/// \brief Feed in one input to a set of modules and concatenate output.
///
/// In order to avoid assumptions about module output dimensions,
/// the concatenation dimension is a parameter.
/// For example, consider two modules with the following outputs:
///     | a b c |    | j k l |
///     | d e f |    | m n o |
///     | g h i |    | p q r |.
/// If the concatenation dimension is 0, the result will be
///     | a b c |
///     | d e f |
///     | g h i |
///     | j k l |
///     | m n o |
///     | p q r |.
/// If the concatenation dimension is 1, the result will be
///     | a b c j k l |
///     | d e f m n o |
///     | g h i p q r |.
/// Modules may not produce square outputs, so often only one dimension will actually work.
/// By default, the last dimension will be used at the concatenation dimension.
template <typename T = double>
class Concat : public Container<T>
{
public:
	using Container<T>::components;
	
	template <typename ... Ms>
	Concat(Module<T> *first, Ms... rest) :
		Container<T>(first, rest...),
		m_concatDim((size_t) -1)
	{}
	
	template <typename ... Ms>
	Concat(size_t concatDim, Ms... components) :
		Container<T>(components...),
		m_concatDim(concatDim)
	{}
	
	Concat(const Concat &module) :
		Container<T>(static_cast<const Container<T> &>(module)),
		m_concatDim(module.m_concatDim)
	{}
	
	Concat(const Serialized &node) :
		Container<T>(node),
		m_concatDim(node.get<size_t>("concatDim"))
	{}
	
	Concat &operator=(const Concat &module)
	{
		Container<T>::operator=(module);
		m_concatDim = module.m_concatDim;
		return *this;
	}
	
	virtual void save(Serialized &node) const override
	{
		Container<T>::save(node);
		node.set("concatDim", m_concatDim);
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		Storage<Tensor<T> *> outputs(components());
		for(size_t i = 0, count = components(); i < count; ++i)
			outputs[i] = &m_components[i]->forward(input);
		
		if(!m_output.sharedWith(outputs))
		{
			m_concatDim = std::min(m_concatDim, outputs[0]->dims() - 1);
			m_output = Tensor<T>::concatenate(outputs, m_concatDim);
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		size_t offset = 0, stride;
		for(size_t i = 0, count = components(); i < count; ++i)
		{
			stride = m_components[i]->output().size(m_concatDim);
			m_components[i]->backward(input, outGrad.narrow(m_concatDim, offset, stride));
			offset += stride;
		}
		
		m_inGrad = m_components[0]->inGrad().copy();
		for(size_t i = 1, count = components(); i < count; ++i)
			m_inGrad.add(m_components[i]->inGrad());
		
		return m_inGrad;
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
	using Container<T>::m_components;
	
private:
	size_t m_concatDim;
};

}

NNRegisterType(Concat, Module);

#endif

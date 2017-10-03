#ifndef NN_MAP_HPP
#define NN_MAP_HPP

#include "module.hpp"

namespace nnlib
{

/// Abstract base class for pointwise functions on inputs, also known as activation functions.
template <typename T = double>
class Map : public Module<T>
{
public:
	/// Single element forward.
	virtual T forwardOne(const T &x) = 0;
	
	/// Single element backward.
	virtual T backwardOne(const T &x, const T &y) = 0;
	
	// MARK: Serialization
	
	virtual void save(Serialized &node) const override
	{}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		m_output.resize(input.shape());
		auto i = input.begin(), j = input.end();
		for(auto k = m_output.begin(); i != j; ++i, ++k)
			*k = forwardOne(*i);
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
		m_inGrad.resize(input.shape());
		auto i = input.begin(), j = input.end(), b = outGrad.begin();
		for(auto k = m_output.begin(), l = m_inGrad.begin(); i != j; ++i, ++b, ++k, ++l)
			*l = *b * backwardOne(*i, *k);
		return m_inGrad;
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(Map, Module);

#endif

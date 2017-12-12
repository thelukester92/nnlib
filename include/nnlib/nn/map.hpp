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
		forEach([&](const T &x, T &y)
		{
			y = forwardOne(x);
		}, input, m_output);
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
		m_inGrad.resize(input.shape());
		forEach([&](const T &x, const T &y, const T &w, T &z)
		{
			z = w * backwardOne(x, y);
		}, input, m_output, outGrad, m_inGrad);
		return m_inGrad;
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(Map, Module);

#endif

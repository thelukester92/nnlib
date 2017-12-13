#ifndef NN_MAP_TPP
#define NN_MAP_TPP

#include "../map.hpp"

namespace nnlib
{

template <typename T>
void Map<T>::save(Serialized &node) const
{}

template <typename T>
Tensor<T> &Map<T>::forward(const Tensor<T> &input)
{
	m_output.resize(input.shape());
	forEach([&](const T &x, T &y)
	{
		y = forwardOne(x);
	}, input, m_output);
	return m_output;
}

template <typename T>
Tensor<T> &Map<T>::backward(const Tensor<T> &input, const Tensor<T> &outGrad)
{
	NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
	m_inGrad.resize(input.shape());
	forEach([&](const T &x, const T &y, const T &w, T &z)
	{
		z = w * backwardOne(x, y);
	}, input, m_output, outGrad, m_inGrad);
	return m_inGrad;
}

}

#endif

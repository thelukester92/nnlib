#ifndef CRITICS_MSE_TPP
#define CRITICS_MSE_TPP

#include "../mse.hpp"

namespace nnlib
{

template <typename T>
MSE<T>::MSE(bool average) :
	m_average(average)
{}

template <typename T>
bool MSE<T>::average() const
{
	return m_average;
}

template <typename T>
MSE<T> &MSE<T>::average(bool ave)
{
	m_average = ave;
	return *this;
}

template <typename T>
T MSE<T>::forward(const Tensor<T> &input, const Tensor<T> &target)
{
	NNAssertEquals(input.shape(), target.shape(), "Incompatible operands!");
	
	auto tar = target.begin();
	T diff, sum = 0;
	forEach([&](const T &inp)
	{
		diff = inp - *tar;
		sum += diff * diff;
		++tar;
	}, input);
	
	if(m_average)
		sum /= input.size();
	
	return sum;
}

template <typename T>
Tensor<T> &MSE<T>::backward(const Tensor<T> &input, const Tensor<T> &target)
{
	NNAssertEquals(input.shape(), target.shape(), "Incompatible operands!");
	m_inGrad.resize(input.shape());
	
	T norm = 2.0;
	if(m_average)
		norm /= input.size();
	
	auto inp = input.begin(), tar = target.begin();
	forEach([&](T &g)
	{
		g = norm * (*inp - *tar);
		++inp;
		++tar;
	}, m_inGrad);
	
	return m_inGrad;
}

}

#endif

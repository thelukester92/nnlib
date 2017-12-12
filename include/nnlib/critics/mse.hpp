#ifndef CRITICS_MSE_HPP
#define CRITICS_MSE_HPP

#include "critic.hpp"

namespace nnlib
{

/// \brief Mean squared error critic.
///
/// When average = false, this is sum squared error.
template <typename T = double>
class MSE : public Critic<T>
{
public:
	MSE(bool average = true) :
		m_average(average)
	{}
	
	bool average() const
	{
		return m_average;
	}
	
	MSE &average(bool ave)
	{
		m_average = ave;
		return *this;
	}
	
	/// L = 1/n sum_i( (input(i) - target(i))^2 )
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override
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
	
	/// dL/di = 2/n (input(i) - target(i))
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
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
	
protected:
	using Critic<T>::m_inGrad;

private:
	bool m_average;
};

}

#endif

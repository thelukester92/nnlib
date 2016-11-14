#ifndef SQUARED_ERROR_H
#define SQUARED_ERROR_H

#include "critic.h"

namespace nnlib
{

/// (Sum) squared error critic.
/// Only makes sense for floating point input/targets.
template <typename T>
class SquaredError : public Critic<T>
{
public:
	SquaredError(size_t inps) : m_loss(inps), m_blame(inps)
	{}
	
	/// Feed in input and target vectors and return a cached error (loss) vector.
	virtual Vector<T> &forward(const Vector<T> &input, const Vector<T> &target) override
	{
		size_t n = input.size();
		Assert(n == m_loss.size(), "Incompatible input to error function!");
		Assert(n == target.size(), "Incompatible operands to error function!");
		m_loss = target - input;
		for(size_t i = 0; i < n; ++i)
			m_loss[i] *= m_loss[i];
		return m_loss;
	}
	
	/// Feed in input and target vectors and return a cached blame (gradient) vector.
	virtual Vector<T> &backward(const Vector<T> &input, const Vector<T> &target) override
	{
		Assert(input.size() == m_loss.size(), "Incompatible input to error function!");
		Assert(input.size() == target.size(), "Incompatible operands to error function!");
		return m_blame = target - input; // technically, this should be multiplied by 2; this is absorbed as a constant.
	}
	
	/// Get the error (loss) buffer.
	virtual Vector<T> &error() override
	{
		return m_loss;
	}
	
	/// Get the blame (gradient) buffer.
	virtual Vector<T> &blame() override
	{
		return m_blame;
	}

private:
	Vector<T> m_loss;
	Vector<T> m_blame;
};

}

#endif

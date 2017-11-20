#ifndef NN_RELU_HPP
#define NN_RELU_HPP

#include <math.h>
#include "map.hpp"

namespace nnlib
{

/// Rectified linear activation function.
template <typename T = double>
class ReLU : public Map<T>
{
public:
	ReLU(T leak = 0.1) :
		m_leak(leak)
	{}
	
	ReLU(const ReLU &module) :
		m_leak(module.m_leak)
	{}
	
	ReLU(const Serialized &node) :
		m_leak(node.get<T>("leak"))
	{}
	
	ReLU &operator=(const ReLU &module)
	{
		m_leak = module.m_leak;
		return *this;
	}
	
	/// Save to a serialized node.
	virtual void save(Serialized &node) const override
	{
		Map<T>::save(node);
		node.set("leak", m_leak);
	}
	
	/// Get the "leak" for this ReLU. 0 if non-leaky.
	T leak() const
	{
		return m_leak;
	}
	
	/// Set the "leak" for this ReLU. 0 <= leak < 1.
	ReLU &leak(T leak)
	{
		NNAssertGreaterThanOrEquals(leak, 0, "Expected positive leak!");
		NNAssertLessThan(leak, 1, "Expected leak to be a percentage!");
		m_leak = leak;
		return *this;
	}
	
	/// Single element forward.
	virtual T forwardOne(const T &x) override
	{
		return x > 0 ? x : m_leak * x;
	}
	
	/// Single element backward.
	virtual T backwardOne(const T &x, const T &y) override
	{
		return x > 0 ? 1 : m_leak;
	}
	
private:
	T m_leak;
};

}

NNRegisterType(ReLU, Module);

#endif

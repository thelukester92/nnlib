#ifndef NN_RELU_H
#define NN_RELU_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Rectified linear activation function.
template <typename T = double>
class ReLU : public Map<T>
{
public:
	using Map<T>::Map;
	using Map<T>::forward;
	using Map<T>::backward;
	
	ReLU(size_t outs = 0, size_t bats = 1) :
		Map<T>(outs, bats),
		m_leak(0.0)
	{}
	
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
	virtual T forward(const T &x) override
	{
		return x > 0 ? x : m_leak * x;
	}
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) override
	{
		return x > 0 ? 1 : m_leak;
	}
	
	/// \brief Write to an archive.
	///
	/// \param ar The archive to which to write.
	template <typename Archive>
	void save(Archive &ar) const
	{
		ar(this->inputs(), m_leak);
	}
	
	/// \brief Read from an archive.
	///
	/// \param ar The archive from which to read.
	template <typename Archive>
	void load(Archive &ar)
	{
		Storage<size_t> shape;
		ar(shape, m_leak);
		this->inputs(shape);
	}
private:
	T m_leak;
};

}

NNRegisterType(ReLU<float>, Module<float>);
NNRegisterType(ReLU<double>, Module<double>);

#endif

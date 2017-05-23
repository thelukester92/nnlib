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
		NNAssert(leak >= 0 && leak < 1, "Invalid parameter for leak! Must be in [0, 1).");
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
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<ReLU>::name << this->inputs();
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(
			str == Binding<ReLU>::name,
			"Unexpected type! Expected '" + Binding<ReLU>::name + "', got '" + str + "'!"
		);
		Storage<size_t> shape;
		in >> shape;
		this->inputs(shape);
	}
	
private:
	T m_leak;
};

NNSerializable(ReLU<double>, Module<double>);
NNSerializable(ReLU<float>, Module<float>);

}

#endif

#ifndef NN_TANH_H
#define NN_TANH_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Hyperbolic tangent activation function.
template <typename T = double>
class TanH : public Map<T>
{
public:
	using Map<T>::Map;
	using Map<T>::forward;
	using Map<T>::backward;
	
	/// Single element forward.
	virtual T forward(const T &x) override
	{
		return tanh(x);
	}
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) override
	{
		return 1.0 - y * y;
	}
	
	// MARK: Serialization
	/*
	/// \brief Write to an archive.
	///
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<TanH>::name << this->inputs();
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(
			str == Binding<TanH>::name,
			"Unexpected type! Expected '" + Binding<TanH>::name + "', got '" + str + "'!"
		);
		Storage<size_t> shape;
		in >> shape;
		this->inputs(shape);
	}
	*/
};

NNSerializable(TanH<double>, Module<double>);
NNSerializable(TanH<float>, Module<float>);

}

#endif

#ifndef NN_LOGISTIC_H
#define NN_LOGISTIC_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Sigmoidal logistic activation function.
template <typename T = double>
class Logistic : public Map<T>
{
public:
	using Map<T>::Map;
	using Map<T>::forward;
	using Map<T>::backward;
	
	/// Single element forward.
	virtual T forward(const T &x) override
	{
		return 1.0 / (1.0 + exp(-x));
	}
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) override
	{
		return y * (1.0 - y);
	}
	
	// MARK: Serialization
	/*
	/// \brief Write to an archive.
	///
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<Logistic>::name << this->inputs();
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(
			str == Binding<Logistic>::name,
			"Unexpected type! Expected '" + Binding<Logistic>::name + "', got '" + str + "'!"
		);
		Storage<size_t> shape;
		in >> shape;
		this->inputs(shape);
	}
	*/
};

}

#endif

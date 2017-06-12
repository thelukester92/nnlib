#ifndef NN_IDENTITY_H
#define NN_IDENTITY_H

#include <math.h>
#include "map.h"

namespace nnlib
{

/// Identity activation function.
/// Useful in a concat module.
template <typename T = double>
class Identity : public Map<T>
{
public:
	using Map<T>::Map;
	using Map<T>::forward;
	using Map<T>::backward;
	
	/// Single element forward.
	virtual T forward(const T &x) override
	{
		return x;
	}
	
	/// Single element backward.
	virtual T backward(const T &x, const T &y) override
	{
		return 1.0;
	}
	
	/// \brief Write to an archive.
	///
	/// \param ar The archive to which to write.
	template <typename Archive>
	void save(Archive &ar) const
	{
		ar(this->inputs());
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	template <typename Archive>
	void load(Archive &in)
	{
		Storage<size_t> shape;
		in >> shape;
		this->inputs(shape);
	}
};

}

#endif

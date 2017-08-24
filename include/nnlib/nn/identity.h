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
	
	/// Save to a serialized node.
	virtual void save(SerializedNode &node) const override {}
	
	/// Load from a serialized node.
	virtual void load(const SerializedNode &node) override {}
	
	/*
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
	/// \param ar The archive from which to read.
	template <typename Archive>
	void load(Archive &ar)
	{
		Storage<size_t> shape;
		ar(shape);
		this->inputs(shape);
	}
	*/
};

}

NNRegisterType(Identity<float>, Module<float>);
NNRegisterType(Identity<double>, Module<double>);

#endif

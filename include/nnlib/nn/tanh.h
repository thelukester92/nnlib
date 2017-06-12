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
};

}

NNRegisterType(TanH<float>, Module<float>);
NNRegisterType(TanH<double>, Module<double>);

#endif

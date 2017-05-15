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
	
	/// \brief A name for this module type.
	///
	/// This may be used for debugging, serialization, etc.
	/// The type should NOT include whitespace.
	static std::string type()
	{
		return "identity";
	}
	
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
};

}

#endif

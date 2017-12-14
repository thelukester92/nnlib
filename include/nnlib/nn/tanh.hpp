#ifndef NN_TANH_HPP
#define NN_TANH_HPP

#include <math.h>
#include "map.hpp"

namespace nnlib
{

/// Hyperbolic tangent activation function.
template <typename T = double>
class TanH : public Map<T>
{
public:
	TanH();
	TanH(const Serialized &);
	TanH(const TanH &);
	TanH &operator=(const TanH &);
	
	virtual T forwardOne(const T &x) override;
	virtual T backwardOne(const T &x, const T &y) override;
};

}

NNRegisterType(TanH, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::TanH<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/tanh.tpp"
#endif

#endif

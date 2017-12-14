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
	ReLU(T leak = 0.1);
	ReLU(const ReLU &module);
	ReLU(const Serialized &node);
	
	ReLU &operator=(const ReLU &module);
	
	virtual void save(Serialized &node) const override;
	
	T leak() const;
	ReLU &leak(T leak);
	
	virtual T forwardOne(const T &x) override;
	virtual T backwardOne(const T &x, const T &y) override;
	
private:
	T m_leak;
};

}

NNRegisterType(ReLU, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::ReLU<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/relu.tpp"
#endif

#endif

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
NNTemplateDefinition(ReLU, "detail/relu.tpp");

#endif

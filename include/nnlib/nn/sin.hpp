#ifndef NN_SIN_HPP
#define NN_SIN_HPP

#include <math.h>
#include <nnlib/nn/map.hpp>

namespace nnlib
{

/// Sinusoid activation function.
template <typename T = double>
class Sin : public Map<T>
{
public:
	Sin()
	Sin(const Serialized &)
	Sin(const Sin &)
	Sin &operator=(const Sin &)
	
	virtual T forwardOne(const T &x) override;
	virtual T backwardOne(const T &x, const T &y) override;
};

}

NNRegisterType(Sin, Module);
NNTemplateDefinition(Sin, "detail/sin.tpp");

#endif

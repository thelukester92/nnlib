#ifndef NN_LOGISTIC_HPP
#define NN_LOGISTIC_HPP

#include <math.h>
#include "map.hpp"

namespace nnlib
{

/// Sigmoidal logistic activation function.
template <typename T = double>
class Logistic : public Map<T>
{
public:
	Logistic();
	Logistic(const Serialized &);
	Logistic(const Logistic &);
	Logistic &operator=(const Logistic &);
	
	virtual T forwardOne(const T &x) override;
	virtual T backwardOne(const T &x, const T &y) override;
};

}

NNRegisterType(Logistic, Module);
NNTemplateDefinition(Logistic, "detail/logistic.tpp");

#endif

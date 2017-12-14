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

#ifdef NN_REAL_T
	extern template class nnlib::Logistic<NN_REAL_T>;
#else
	#include "detail/logistic.tpp"
#endif

#endif

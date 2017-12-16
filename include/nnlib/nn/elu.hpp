#ifndef NN_ELU_HPP
#define NN_ELU_HPP

#include <math.h>
#include <nnlib/nn/map.hpp>

namespace nnlib
{

/// Exponential linear activation function.
template <typename T = NN_REAL_T>
class ELU : public Map<T>
{
public:
	ELU(T alpha = 1.0);
	ELU(const ELU &module);
	ELU(const Serialized &node);
	
	ELU &operator=(const ELU &module);
	
	virtual void save(Serialized &node) const override;
	
	T alpha() const;
	ELU &alpha(T alpha);
	
	virtual T forwardOne(const T &x) override;
	virtual T backwardOne(const T &x, const T &y) override;
	
private:
	T m_alpha;
};

}

NNRegisterType(ELU, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::ELU<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/elu.tpp"
#endif

#endif

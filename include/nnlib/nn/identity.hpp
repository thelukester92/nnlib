#ifndef NN_IDENTITY_HPP
#define NN_IDENTITY_HPP

#include "module.hpp"

namespace nnlib
{

/// \brief Identity "map" for implementing resisual connections.
///
/// A residual connection can be modeled for an arbitrary module m like this:
///     residual = new Concat<T>(new Identity<T>(), m);
template <typename T = double>
class Identity : public Module<T>
{
public:
	Identity();
	Identity(const Serialized &);
	Identity(const Identity &);
	Identity &operator=(const Identity &);
	
	virtual void save(Serialized &node) const override;
	virtual Tensor<T> &forward(const Tensor<T> &input) override;
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override;
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(Identity, Module);

#if defined NN_REAL_T && !defined NN_IMPL
	extern template class nnlib::Identity<NN_REAL_T>;
#elif !defined NN_IMPL
	#include "detail/identity.tpp"
#endif

#endif

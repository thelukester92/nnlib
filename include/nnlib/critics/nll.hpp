#ifndef CRTIICS_NLL_HPP
#define CRTIICS_NLL_HPP

#include "critic.hpp"

namespace nnlib
{

/// \brief Negative log loss critic.
///
/// This critic requires matrix input and single-column matrix output.
template <typename T = double>
class NLL : public Critic<T>
{
public:
	NLL(bool average = true);
	
	bool average() const;
	NLL &average(bool ave);
	
	/// A convenience method for counting misclassifications, since we know the output will be categorical.
	size_t misclassifications(const Tensor<T> &input, const Tensor<T> &target);
	
	/// L = 1/n sum_i( -input(target(i)) )
	virtual T forward(const Tensor<T> &input, const Tensor<T> &target) override;
	
	/// dL/di = target == i ? -1/n : 0
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override;
	
protected:
	using Critic<T>::m_inGrad;
	
private:
	bool m_average;
};

}

#ifdef NN_REAL_T
	extern template class nnlib::NLL<NN_REAL_T>;
#else
	#include "detail/nll.tpp"
#endif

#endif

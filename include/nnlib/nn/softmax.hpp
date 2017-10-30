#ifndef NN_SOFT_MAX_HPP
#define NN_SOFT_MAX_HPP

#include "module.hpp"

namespace nnlib
{

/// \brief Softmax for classification problems.
///
/// When using NLL, LogSoftMax is preferred.
template <typename T = double>
class SoftMax : public Module<T>
{
public:
	SoftMax() {}
	SoftMax(const SoftMax &module) {}
	SoftMax(const Serialized &node) {}
	SoftMax &operator=(const SoftMax &module) { return *this; }
	
	// MARK: Serialization
	
	virtual void save(Serialized &node) const override {}
	
	// MARK: Computation
	
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		m_output.resize(input.shape());
		
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			T max = input.narrow(0, i).max(), sum = 0;
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				sum += (m_output(i, j) = exp(input(i, j) - max));
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				m_output(i, j) /= sum;
		}
		
		return m_output;
	}
	
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssertEquals(input.dims(), 2, "Expected matrix input!");
		NNAssertEquals(input.shape(), m_output.shape(), "LogSoftMax::forward must be called first!");
		NNAssertEquals(input.shape(), outGrad.shape(), "Incompatible input and outGrad!");
		m_inGrad.resize(input.shape());
		
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			T sum = 0;
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				sum += outGrad(i, j) * m_output(i, j);
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				m_inGrad(i, j) = m_output(i, j) * (outGrad(i, j) - sum);
		}
		
		return m_inGrad;
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(SoftMax, Module);

#endif

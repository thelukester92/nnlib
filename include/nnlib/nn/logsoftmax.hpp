#ifndef NN_LOG_SOFT_MAX_HPP
#define NN_LOG_SOFT_MAX_HPP

#include "module.hpp"

namespace nnlib
{

/// \brief Log softmax for classification problems.
///
/// Best combined with NLL critic. Works only on 2D tensors.
template <typename T = double>
class LogSoftMax : public Module<T>
{
public:
	LogSoftMax() {}
	LogSoftMax(const LogSoftMax &module) {}
	LogSoftMax(const Serialized &node) {}
	LogSoftMax &operator=(const LogSoftMax &module) { return *this; }
	
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
				sum += exp(input(i, j) - max);
			sum = max + log(sum);
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				m_output(i, j) = input(i, j) - sum;
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
			T sum = outGrad.narrow(0, i).sum();
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
				m_inGrad(i, j) = outGrad(i, j) - exp(m_output(i, j)) * sum;
		}
		
		return m_inGrad;
	}
	
protected:
	using Module<T>::m_output;
	using Module<T>::m_inGrad;
};

}

NNRegisterType(LogSoftMax, Module);

#endif

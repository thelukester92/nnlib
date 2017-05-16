#ifndef NN_LOG_SOFT_MAX_H
#define NN_LOG_SOFT_MAX_H

#include "module.h"

namespace nnlib
{

/// Log soft max module for classification problems.
template <typename T = double>
class LogSoftMax : public Module<T>
{
public:
	using Module<T>::inputs;
	using Module<T>::outputs;
	
	LogSoftMax(size_t outs = 0, size_t bats = 1) :
		m_inGrad(bats, outs),
		m_output(bats, outs)
	{}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		NNAssert(input.dims() == 2, "LogSoftMax expects Matrix input!");
		
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			T max = input.narrow(0, i).max(), sum = 0;
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
			{
				sum += exp(input(i, j) - max);
			}
			sum = max + log(sum);
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
			{
				m_output(i, j) = input(i, j) - sum;
			}
		}
		
		return m_output;
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		NNAssert(input.dims() == 2, "Linear expects Matrix input!");
		NNAssert(outGrad.dims() == 2, "Linear expects Matrix output gradient!");
		
		for(size_t i = 0, iend = input.size(0); i < iend; ++i)
		{
			T sum = outGrad.narrow(0, i).sum();
			for(size_t j = 0, jend = input.size(1); j < jend; ++j)
			{
				m_inGrad(i, j) = outGrad(i, j) - exp(m_output(i, j)) * sum;
			}
		}
		
		return m_inGrad;
	}
	
	/// Cached output.
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Cached input gradient.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}
	
	/// Set the input shape of this module, including batch.
	/// In LogSoftMax, input shape is always equal to output shape.
	virtual LogSoftMax &inputs(const Storage<size_t> &dims) override
	{
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		return *this;
	}
	
	/// Set the output shape of this module, including batch.
	/// In LogSoftMax, input shape is always equal to output shape.
	virtual LogSoftMax &outputs(const Storage<size_t> &dims) override
	{
		Module<T>::inputs(dims);
		Module<T>::outputs(dims);
		return *this;
	}
	
	// MARK: Serialization
	
	/// \brief Write to an archive.
	///
	/// \param out The archive to which to write.
	virtual void save(Archive &out) const override
	{
		out << Binding<LogSoftMax>::name;
	}
	
	/// \brief Read from an archive.
	///
	/// \param in The archive from which to read.
	virtual void load(Archive &in) override
	{
		std::string str;
		in >> str;
		NNAssert(
			str == Binding<LogSoftMax>::name,
			"Unexpected type! Expected '" + Binding<LogSoftMax>::name + "', got '" + str + "'!"
		);
	}
	
private:
	Tensor<T> m_inGrad;	///< Input gradient buffer.
	Tensor<T> m_output;	///< Output buffer.
};

NNSerializable(LogSoftMax<double>, Module<double>);
NNSerializable(LogSoftMax<float>, Module<float>);

}

#endif

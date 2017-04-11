#ifndef POOL_H
#define POOL_H

#include "../module.h"

namespace nnlib
{

/// A module that pools inputs (using different kinds of reductions like product or sum).
/// F must have static forward and backward functions.
template <template <typename> class F, typename T = double>
class Pool : public Module<T>
{
public:
	Pool(size_t inps = 0, size_t slices = 2, size_t batch = 1) :
		m_inputBlame(batch, inps),
		m_outputs(batch, inps / slices)
	{
		NNAssert(inps % slices == 0, "Invalid number of slices for the given number of inputs!");
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == this->batchSize() && inputs.cols() == this->inputs(), "Incompatible input!");
		/// \todo forward pooling using the number of slices
		/// \todo delete below code
		/// \todo also make a virtual Matrix<T> &forward(const Table<T> &inputs) for flattening
		
		
		auto i = inputs.begin();
		auto j = m_outputs.begin(), end = m_outputs.end();
		for(; j != end; ++i, ++j)
			*j = *i;
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(blame.rows() == m_inputBlame.rows() && blame.cols() == m_inputBlame.cols(), "Incompatible blame!");
		auto i = blame.begin();
		auto j = m_inputBlame.begin(), end = m_inputBlame.end();
		for(; j != end; ++i, ++j)
			*j = *i;
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_outputs;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
private:
	Matrix<T> m_inputBlame;	///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_outputs;	///< The output of this layer.
};

}

#endif

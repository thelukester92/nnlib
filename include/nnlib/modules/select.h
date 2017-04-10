#ifndef SELECT_H
#define SELECT_H

#include "../module.h"

namespace nnlib
{

/// Selects a portion of the input to pass through.
template <typename T = double>
class Select : public Module<T>
{
public:
	Select(size_t offset, size_t inps, size_t outs, size_t batch = 1) :
		m_offset(offset),
		m_inputBlame(batch, inps),
		m_output(batch, outs)
	{}
	
	Select(size_t offset, size_t outs) :
		m_offset(offset),
		m_inputBlame(1, 0),
		m_output(1, outs)
	{}
	
	Select &offset(size_t offset)
	{
		NNAssert(this->outputs() + offset <= this->inputs(), "Invalid offset and length for a Select module!");
		m_offset = offset;
	}
	
	virtual void resize(size_t inps, size_t outs) override
	{
		NNAssert(outs + m_offset <= inps, "Invalid offset and length for a Select module!");
		Module<T>::resize(inps, outs);
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &input) override
	{
		NNAssert(input.rows() == m_inputBlame.rows(), "Invalid batch size!");
		NNAssert(input.cols() == m_inputBlame.cols(), "Invalid input size!");
		
		for(size_t row = 0, rend = input.rows(); row < rend; ++row)
			for(size_t i = 0, iend = m_output.cols(); i < iend; ++i)
				m_output(row, i) = input(row, i + m_offset);
		
		return m_output;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &input, const Matrix<T> &blame) override
	{
		NNAssert(input.rows() == m_inputBlame.rows(), "Invalid batch size!");
		NNAssert(input.cols() == m_inputBlame.cols(), "Invalid input size!");
		NNAssert(blame.rows() == m_output.rows(), "Invalid batch size!");
		NNAssert(blame.cols() == m_output.cols(), "Invalid output size!");
		
		m_inputBlame.fill(0.0);
		for(size_t row = 0, rend = input.rows(); row < rend; ++row)
			for(size_t i = 0, iend = m_output.cols(); i < iend; ++i)
				m_inputBlame(row, i + m_offset) = blame(row, i);
		
		return m_inputBlame;
	}
	
	virtual Matrix<T> &output() override
	{
		return m_output;
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_inputBlame;
	}
	
private:
	size_t m_offset;		///< The number of inputs to skip.
	Matrix<T> m_inputBlame;	///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_output;		///< The output of this layer.
};

}

#endif

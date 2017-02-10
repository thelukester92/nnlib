#ifndef ALL_PAIRS_H
#define ALL_PAIRS_H

#include "module.h"

namespace nnlib
{

/// An activation function layer that outputs all N(N-1)/2 pairs of inputs.
template <typename T = double>
class AllPairs : public Module<T>
{
public:
	AllPairs(size_t inps = 0, size_t batch = 1) : m_inputBlame(batch, inps), m_outputs(batch, inps * (inps - 1))
	{}
	
	virtual void resize(size_t inps, size_t outs, size_t bats) override
	{
		NNAssert(outs == inps * (inps - 1), "AllPairs must have N * (N-1) outputs!");
		Module<T>::resize(inps, outs, bats);
	}
	
	virtual void resize(size_t inps) override
	{
		resize(inps, inps * (inps - 1), m_outputs.rows());
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		NNAssert(inputs.rows() == m_inputBlame.rows() && inputs.cols() == m_inputBlame.cols(), "Incompatible input!");
		auto k = m_outputs.begin();
		
		for(size_t row = 0; row < m_outputs.rows(); ++row)
		{
			for(size_t i = 0; i < inputs.cols(); ++i)
			{
				for(size_t j = i + 1; j < inputs.cols(); ++j)
				{
					*k = inputs(row, i); ++k;
					*k = inputs(row, j); ++k;
				}
			}
		}
		
		return m_outputs;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		NNAssert(blame.rows() == m_outputs.rows() && blame.cols() == m_outputs.cols(), "Incompatible blame!");
		m_inputBlame.fill(0);
		auto k = blame.begin();
		
		for(size_t row = 0; row < blame.rows(); ++row)
		{
			for(size_t i = 0; i < m_inputBlame.cols(); ++i)
			{
				for(size_t j = i + 1; j < m_inputBlame.cols(); ++j)
				{
					m_inputBlame(row, i) += *k;
					m_inputBlame(row, j) += *k;
					++k;
				}
			}
		}
		
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

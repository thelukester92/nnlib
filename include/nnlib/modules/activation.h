#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "../module.h"

namespace nnlib
{

/// A module that applies an activation function to each input.
/// F must have static forward and backward functions.
template <template <typename> class F, typename T = double>
class Activation : public Module<T>
{
public:
	Activation(size_t size = 0, size_t batch = 1) :
		m_inputBlame(batch, size),
		m_output(batch, size)
	{}
	
	virtual void resize(size_t inps) override
	{
		Module<T>::resize(inps, inps);
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &input) override
	{
		NNAssert(input.rows() == this->batchSize(), "Incorrect batch size!");
		NNAssert(input.cols() == this->inputs(), "Incorrect input size!");
		
		auto i = input.begin();
		auto j = m_output.begin(), end = m_output.end();
		for(; j != end; ++i, ++j)
			*j = F<T>::forward(*i);
		
		return m_output;
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &input, const Matrix<T> &blame) override
	{
		NNAssert(input.rows() == this->batchSize(), "Incorrect batch size!");
		NNAssert(input.cols() == this->inputs(), "Incorrect input size!");
		NNAssert(blame.rows() == this->batchSize(), "Incorrect batch size!");
		NNAssert(blame.cols() == this->outputs(), "Incorrect blame size!");
		
		auto i = input.begin();
		auto j = m_output.begin();
		auto k = blame.begin();
		auto l = m_inputBlame.begin(), end = m_inputBlame.end();
		for(; l != end; ++i, ++j, ++k, ++l)
			*l = *k * F<T>::backward(*i, *j);
		
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
	Matrix<T> m_inputBlame;	///< Gradient of the error w.r.t. the inputs.
	Matrix<T> m_output;		///< The output of this layer.
};

}

#endif

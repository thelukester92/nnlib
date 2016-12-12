#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include "module.h"

namespace nnlib
{

template <typename T>
class Linear : public Module<T>
{
public:
	Linear(size_t inps, size_t outs, size_t batch) : m_weights(outs, inps), m_outputs(batch, outs), m_weightsBlame(outs, inps), m_inputBlame(batch, inps) {}
	
	virtual void forward(const Matrix<T> &inputs) override
	{
		std::cout << inputs.rows() << "x" << inputs.cols() << " * " << m_weights.cols() << "x" << m_weights.rows() << " = " << m_outputs.rows() << "x" << m_outputs.cols() << std::endl;
		Matrix<T>::multiply(inputs, m_weights, m_outputs, false, true);
	}
	
	virtual void backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		Matrix<T>::multiply(blame, inputs, m_weightsBlame, true, false);
		Matrix<T>::multiply(blame, m_weights, m_inputBlame);
	}
	
	virtual Vector<Matrix<T> *> parameters() override
	{
		return { &m_weights };
	}
	
	virtual Vector<Matrix<T> *> blame() override
	{
		return { &m_weightsBlame };
	}
	
private:
	Matrix<T> m_weights;
	Matrix<T> m_outputs;
	Matrix<T> m_weightsBlame;
	Matrix<T> m_inputBlame;
};

}

#endif

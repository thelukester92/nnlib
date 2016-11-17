#ifndef MODULE_H
#define MODULE_H

#include "tensor.h"

namespace nnlib
{

/// \todo what if the module feeds forward a Matrix instead of a Vector?
/// \todo Matrix-Matrix feed-in and Matrix-Vector feed-in.

template <typename T>
class Module
{
public:
	Module(size_t inps, size_t outs, size_t batchSize)
	: m_inputBlame(batchSize, inps), m_output(batchSize, outs)
	{}
	
	virtual ~Module() {}
	
	size_t inputCount() const
	{
		return m_inputBlame.cols();
	}
	
	size_t outputCount() const
	{
		return m_output.cols();
	}
	
	Matrix<T> &inputBlame()
	{
		return m_inputBlame;
	}
	
	const Matrix<T> &inputBlame() const
	{
		return m_inputBlame;
	}
	
	Matrix<T> &output()
	{
		return m_output;
	}
	
	const Matrix<T> &output() const
	{
		return m_output;
	}
	
	/// Resize this module.
	virtual void resize(size_t inps, size_t outs, size_t batchSize)
	{
		m_inputBlame.resize(batchSize, inps);
		m_output.resize(batchSize, outs);
	}
	
	/// Feed in a single input vector and return the output vector.
	virtual Vector<T> forward(const Vector<T> &input)
	{
		forward(Matrix<T>(input, 1, input.size()));
		return m_output.row(0);
	}
	
	/// Feed in input vectors and return cached output vectors.
	virtual Matrix<T> &forward(const Matrix<T> &input) = 0;
	
	/// Feed in inputs and output blames (gradient) and return cached input blame vectors.
	virtual Matrix<T> &backward(const Matrix<T> &input, const Matrix<T> &blame) = 0;
	
	/// Return pointers to all parameters (i.e. for flattening).
	virtual Vector<Tensor<T> *> parameters()
	{
		return Vector<Tensor<T> *>(0);
	}
	
	/// Return pointers to parameter blame buffers (i.e. for flattening).
	virtual Vector<Tensor<T> *> blame()
	{
		return Vector<Tensor<T> *>(0);
	}

protected:
	Matrix<T> m_inputBlame;
	Matrix<T> m_output;
};

}

#endif

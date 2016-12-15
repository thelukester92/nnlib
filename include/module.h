#ifndef MODULE_H
#define MODULE_H

#include <iostream>

namespace nnlib
{

template <typename T = double>
class Module
{
public:
	typedef T type;
	
	virtual ~Module() {}
	
	/// Change the input, output, and batch size.
	virtual void resize(size_t inps, size_t outs, size_t bats)
	{
		output().resize(bats, outs);
		inputBlame().resize(bats, inps);
	}
	
	/// Change the input and output size.
	virtual void resize(size_t inps, size_t outs)
	{
		resize(inps, outs, output().rows());
	}
	
	/// Change the input size.
	virtual void resize(size_t inps)
	{
		resize(inps, output().cols(), output().rows());
	}
	
	/// Change the batch size.
	virtual void batch(size_t size)
	{
		resize(inputBlame().cols(), output().cols(), size);
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) = 0;
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) = 0;
	
	virtual Matrix<T> &output() = 0;
	virtual Matrix<T> &inputBlame() = 0;
	
	virtual Vector<Tensor<T> *> parameters()
	{
		return Vector<Tensor<T> *>();
	}
	
	virtual Vector<Tensor<T> *> blame()
	{
		return Vector<Tensor<T> *>();
	}
};

}

#endif

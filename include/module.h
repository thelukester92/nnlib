#ifndef MODULE_H
#define MODULE_H

namespace nnlib
{

template <typename T>
class Module
{
public:
	typedef T type;
	
	virtual ~Module() {}
	
	/// Change the batch size.
	virtual void batch(size_t size)
	{
		output().resize(size, output().cols());
		inputBlame().resize(size, inputBlame().cols());
	}
	
	/// Change the batch size based on an input.
	virtual void batchFor(const Matrix<T> &inputs)
	{
		batch(inputs.rows());
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

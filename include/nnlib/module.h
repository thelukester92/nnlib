#ifndef MODULE_H
#define MODULE_H

namespace nnlib
{

template <typename T = double>
class Module
{
public:
	virtual Module &resize(size_t inps, size_t outs)
	{
		output().resize(batchSize(), outs);
		inputBlame().resize(batchSize(), inps);
	}
	
	virtual Module &resize(size_t inps)
	{
		resize(inps, outputs());
	}
	
	virtual Module &batch(size_t bats)
	{
		output().resize(bats, outputs());
		inputBlame().resize(bats, inputs());
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) = 0;
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) = 0;
	
	virtual Matrix<T> &output() = 0;
	virtual Matrix<T> &inputBlame() = 0;
	
	size_t inputs() const
	{
		return inputBlame().cols();
	}
	
	size_t outputs() const
	{
		return output().cols();
	}
	
	size_t batchSize() const
	{
		return output().rows();
	}
};

}

#endif

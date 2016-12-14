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

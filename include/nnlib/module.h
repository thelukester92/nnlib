#ifndef MODULE_H
#define MODULE_H

namespace nnlib
{

template <typename T = double>
class Module
{
public:
	virtual void forward(const Tensor<T> &inputs) = 0;
	virtual void backward(const Tensor<T> &inputs, const Tensor<T> &blame) = 0;
	
private:
	
};

}

#endif

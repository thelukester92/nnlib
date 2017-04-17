#ifndef MODULE_H
#define MODULE_H

namespace nnlib
{

/// The abtract base class for all neural network modules.
template <typename T = double>
class Module
{
public:
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) = 0;
	
	/// Backward propagate input and output blame, returning input blame.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outBlame) = 0;
	
	/// Cached output.
	virtual Tensor<T> &output() = 0;
	
	/// Cached input blame.
	virtual Tensor<T> &inBlame() = 0;
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Tensor<Tensor<T> *> &parameters()
	{
		return Tensor<Tensor<T> *>();
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' blame.
	virtual Tensor<Tensor<T> *> &blame()
	{
		return Tensor<Tensor<T> *>();
	}
private:
	
};

}

#endif

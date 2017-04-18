#ifndef NN_MODULE_H
#define NN_MODULE_H

#include "../util/tensor.h"

namespace nnlib
{

/// The abtract base class for all neural network modules.
template <typename T = double>
class Module
{
public:
	virtual ~Module() {}
	
	/// Change the input dimensions of this module.
	virtual void resizeInput(const Storage<size_t> &dims)
	{
		inBlame().resize(dims);
	}
	
	/// Change the input dimensions of this module.
	template <typename ... Ts>
	void resizeInput(Ts... dims)
	{
		resizeInput({ static_cast<size_t>(dims)... });
	}
	
	/// Change the output dimensions of this module.
	virtual void resizeOutput(const Storage<size_t> &dims)
	{
		output().resize(dims);
	}
	
	/// Change the output dimensions of this module.
	template <typename ... Ts>
	void resizeOutput(Ts... dims)
	{
		resizeOutput({ static_cast<size_t>(dims)... });
	}
	
	/// Change both the input and output dimensions of this module.
	virtual void resize(const Storage<size_t> &inDims, const Storage<size_t> &outDims)
	{
		resizeInput(inDims);
		resizeOutput(outDims);
	}
	
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) = 0;
	
	/// Backward propagate input and output blame, returning input blame.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outBlame) = 0;
	
	/// Cached output.
	virtual Tensor<T> &output() = 0;
	
	/// Cached input blame.
	virtual Tensor<T> &inBlame() = 0;
	
	/// A vector of tensors filled with (views of) this module's parameters.
	virtual Storage<Tensor<T> *> parameters()
	{
		return {};
	}
	
	/// A vector of tensors filled with (views of) this module's parameters' blame.
	virtual Storage<Tensor<T> *> blame()
	{
		return {};
	}
};

}

#endif

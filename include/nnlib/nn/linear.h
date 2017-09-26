#ifndef NN_LINEAR_H
#define NN_LINEAR_H

#include "module.h"

namespace nnlib
{

/// A standard feed-forward layer that returns a linear combination of inputs.
template <typename T = double>
class Linear : public Module<T>
{
public:
	// MARK: Serialization
	
	virtual void save(Serialized &) override
	{
		
	}
	
	virtual void load(const Serialized &) override
	{
		
	}
	
	// MARK: Computation
	
	virtual void updateOutput(const Tensor<T> &input) override
	{
		
	}
	
	virtual void updateInGrad(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		
	}
	
	virtual void updateParamGrad(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		
	}
	
	// MARK: Buffers
	
	virtual Storage<Tensor<T> *> paramsList() override
	{
		return {};
	}
	
	virtual Storage<Tensor<T> *> gradList() override
	{
		return {};
	}
};

}

NNRegisterType(Linear, Module);

#endif

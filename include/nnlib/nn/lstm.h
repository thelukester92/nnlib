#ifndef NN_LSTM_H
#define NN_LSTM_H

#include "concat.h"
#include "container.h"
#include "sequential.h"
#include "linear.h"
#include "logistic.h"
#include "tanh.h"

namespace nnlib
{

/// LSTM recurrent module.
template <typename T = double>
class LSTM : public Container<T>
{
public:
	/// Forward propagate input, returning output.
	virtual Tensor<T> &forward(const Tensor<T> &input) override
	{
		
	}
	
	/// Backward propagate input and output gradient, returning input gradient.
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &outGrad) override
	{
		
	}
private:
	
};

}

#endif

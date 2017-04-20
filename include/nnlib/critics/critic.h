#ifndef CRITIC_H
#define CRITIC_H

#include "../util/tensor.h"

namespace nnlib
{

template <typename T = double>
class Critic
{
public:
	/// Calculate the loss (how far input is from target).
	virtual Tensor<T> &forward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Calculate the "blame" (the gradient of the loss w.r.t. the input).
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) = 0;
	
	/// Output buffer (the loss).
	virtual Tensor<T> &output() = 0;
	
	/// Input blame buffer (the gradient).
	virtual Tensor<T> &inBlame() = 0;
};

}

#endif

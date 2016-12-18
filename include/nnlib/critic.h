#ifndef CRITIC_H
#define CRITIC_H

#include "matrix.h"

namespace nnlib
{

template <typename T = double>
class Critic
{
public:
	virtual void batch(size_t size)
	{
		output().resize(size, output().cols());
		blame().resize(size, blame().cols());
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs, const Matrix<T> &targets) = 0;
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &targets) = 0;
	
	virtual Matrix<T> &output() = 0;
	virtual Matrix<T> &blame() = 0;
};

}

#endif

#ifndef TENSOR_UTIL_H
#define TENSOR_UTIL_H

#include "../tensor.h"

namespace nnlib
{

/// Output a tensor to a stream.
template <typename T>
std::ostream &operator<<(std::ostream &out, const Tensor<T> &t)
{
	out << std::left << std::setprecision(5) << std::fixed;
	
	if(t.dims() == 1)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			out << t(i) << "\n";
		}
	}
	else if(t.dims() == 2)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			for(size_t j = 0; j < t.size(1); ++j)
			{
				out << std::setw(10) << t(i, j);
			}
			out << "\n";
		}
	}
	
	out << "[ Tensor of dimension " << t.size(0);
	for(size_t i = 1; i < t.dims(); ++i)
	{
		out << " x " << t.size(i);
	}
	out << " ]";
	
	return out;
}

/// Operator overload for addition assignment.
template <typename T>
Tensor<T> &operator+=(Tensor<T> &lhs, const Tensor<T> &rhs)
{
	return lhs.add(rhs);
}

/// Operator overload for addition.
template <typename T>
Tensor<T> operator+(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
	Tensor<T> sum = lhs.copy();
	return sum += rhs;
}

/// Operator overload for subtraction assignment.
template <typename T>
Tensor<T> &operator-=(Tensor<T> &lhs, const Tensor<T> &rhs)
{
	return lhs.add(rhs, -1);
}

/// Operator overload for subtraction.
template <typename T>
Tensor<T> operator-(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
	Tensor<T> difference = lhs.copy();
	return difference -= rhs;
}

/// Operator overload for primitive multiplication assignment.
template <typename T, typename U>
Tensor<T> &operator*=(Tensor<T> &lhs, const U &rhs)
{
	return lhs.scale(rhs);
}

/// Operator overload for primitive multiplication.
template <typename T, typename U>
Tensor<T> operator*(const Tensor<T> &lhs, const U &rhs)
{
	Tensor<T> product = lhs.copy();
	return product *= rhs;
}

/// Operator overload for primitive division assignment.
template <typename T, typename U>
Tensor<T> &operator/=(Tensor<T> &lhs, const U &rhs)
{
	return lhs.scale(1.0 / rhs);
}

/// Operator overload for primitive division.
template <typename T, typename U>
Tensor<T> operator/(const Tensor<T> &lhs, const U &rhs)
{
	Tensor<T> quotient = lhs.copy();
	return quotient /= rhs;
}

}

#endif

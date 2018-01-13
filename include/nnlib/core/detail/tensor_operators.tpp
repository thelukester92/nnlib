#ifndef CORE_TENSOR_OPERATORS_TPP
#define CORE_TENSOR_OPERATORS_TPP

#include "../tensor.hpp"

#ifndef NN_MAX_NUM_DIMENSIONS
#define NN_MAX_NUM_DIMENSIONS 32ul
#endif

template <typename T>
std::ostream &operator<<(std::ostream &out, const nnlib::Tensor<T> &t)
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

template <typename T>
nnlib::Tensor<T> &operator+=(nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
	forEach([&](T x, T &y)
	{
		y += x;
	}, rhs, lhs);
	return lhs;
}

template <typename T>
nnlib::Tensor<T> operator+(const nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
	nnlib::Tensor<T> sum = lhs.copy();
	return sum += rhs;
}

template <typename T>
nnlib::Tensor<T> &operator-=(nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
	forEach([&](T x, T &y)
	{
		y -= x;
	}, rhs, lhs);
	return lhs;
}

template <typename T>
nnlib::Tensor<T> operator-(const nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
	nnlib::Tensor<T> difference = lhs.copy();
	return difference -= rhs;
}

template <typename T>
nnlib::Tensor<T> &operator*=(nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
	return lhs.scale(rhs);
}

template <typename T>
nnlib::Tensor<T> operator*(const nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
	nnlib::Tensor<T> product = lhs.copy();
	return product *= rhs;
}

template <typename T>
nnlib::Tensor<T> &operator/=(nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
	return lhs.scale(1.0 / rhs);
}

template <typename T>
nnlib::Tensor<T> operator/(const nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
	nnlib::Tensor<T> quotient = lhs.copy();
	return quotient /= rhs;
}

#endif

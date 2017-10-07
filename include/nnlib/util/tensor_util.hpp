#ifndef UTIL_TENSOR_UTIL_HPP
#define UTIL_TENSOR_UTIL_HPP

#include "../core/tensor.hpp"

namespace nnlib
{

/// \brief Sparsify a dense tensor, dropping values with magnitude less than epsilon.
///
/// The output will be a matrix. The number of rows will be the number of non-zero elements;
/// the number of columns will be D + 1 where D is the number of dimensions in the dense tensor.
/// The first D columns in a row are the indices and the last is the value in that slot.
/// The first row in the output will be the sizes of each dimension with one unused column.
///
/// For example, a truncated identity matrix of size 3x5 could be represented like this:
///
///     3 5 0.0   <-- size
///     0 0 1.0
///     1 1 1.0
///     2 2 1.0
template <typename T>
Tensor<T> sparsify(const Tensor<T> &t, T epsilon = 1e-12)
{
	size_t count = 0;
	for(auto x : t)
		if(std::abs(x) > epsilon)
			++count;
	
	Tensor<T> sparse(count, t.dims() + 1);
	
	size_t idx = 0;
	for(auto i = t.begin(), end = t.end(); i != end; ++i)
	{
		if(std::abs(*i) > epsilon)
		{
			for(size_t j = 0, jend = i.indices().size(); j != jend; ++j)
				sparse(idx, j) = i.indices()(j);
			sparse(idx, i.indices().size()) = *i;
			++idx;
		}
	}
	
	return sparse;
}

/// \brief Unsparsify a sparse tensor.
///
/// See sparsify for an explanation of sparse tensors.
template <typename T>
Tensor<T> unsparsify(const Tensor<T> &t)
{
	NNAssertEquals(t.dims(), 2, "Sparse tensors must be represented by matrices!");
	
	Storage<size_t> dims(t.size(1) - 1);
	for(size_t i = 0, end = dims.size(); i != end; ++i)
		dims[i] = t(0, i);
	
	Tensor<T> dense(dims, true);
	dense.fill(0);
	
	for(size_t i = 1, end = t.size(0), jend = t.size(1) - 1; i != end; ++i)
	{
		for(size_t j = 0; j != jend; ++j)
			dims[j] = t(i, j);
		dense(dims) = t(i, jend);
	}
	
	return dense;
}

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

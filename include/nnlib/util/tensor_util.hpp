#ifndef UTIL_TENSOR_UTIL_HPP
#define UTIL_TENSOR_UTIL_HPP

#include "../core/tensor.hpp"

#ifndef NN_MAX_NUM_DIMENSIONS
#define NN_MAX_NUM_DIMENSIONS 32ul
#endif

namespace nnlib
{

namespace detail
{
	template <size_t D, size_t I>
	struct ForEachHelper
	{
		template <typename F, typename ... Ts>
		static void apply(Storage<size_t> &indices, const Storage<size_t> &shape, const Storage<size_t> &strides, F func, Ts &...ts)
		{
			for(indices[I] = 0; indices[I] < shape[I]; ++indices[I])
				ForEachHelper<D-1, I+1>::apply(indices, shape, strides, func, ts...);
		}
	};
	
	template <size_t I>
	struct ForEachHelper<1ul, I>
	{
		template <typename F, typename ... Ts>
		static void apply(Storage<size_t> &indices, const Storage<size_t> &shape, const Storage<size_t> &strides, F func, Ts &...ts)
		{
			for(indices[I] = 0; indices[I] < shape[I]; ++indices[I])
				func(ts.ptr()[indexOf(indices, strides)]...);
		}
		
	private:
		static size_t indexOf(const Storage<size_t> &indices, const Storage<size_t> &strides)
		{
			size_t i = 0, j;
			for(j = 0; j < I + 1; ++j)
				i += indices[j] * strides[j];
			return i;
		}
	};
	
	template <size_t D>
	struct ForEach
	{
		template <typename F, typename ... Ts>
		static void apply(F func, const Storage<size_t> &shape, const Storage<size_t> &strides, Ts &...ts)
		{
			Storage<size_t> indices(D);
			ForEachHelper<D, 0>::apply(indices, shape, strides, func, ts...);
		}
	};
	
	template <size_t MIN, size_t MAX, template <size_t> class WORKER>
	struct TemplateSearch
	{
		template <typename ... Ts>
		static void apply(size_t v, Ts &&...ts)
		{
			if(v == MIN)
				WORKER<MIN>::apply(std::forward<Ts>(ts)...);
			else
				TemplateSearch<MIN+1, MAX, WORKER>::apply(v, std::forward<Ts>(ts)...);
		}
	};
	
	template <size_t MAX, template <size_t> class WORKER>
	struct TemplateSearch<MAX, MAX, WORKER>
	{
		template <typename ... Ts>
		static void apply(size_t v, Ts &&...ts)
		{
			NNHardAssert(v == MAX, "Too many dimensions! Define NN_MAX_NUM_DIMENSIONS to increase this limit.");
			WORKER<MAX>::apply(std::forward<Ts>(ts)...);
		}
	};
}

/// A more efficient way apply a function to each element in one or more tensors.
template <typename F, typename T, typename ... Ts>
void forEach(F func, T &first, Ts &...ts)
{
	detail::TemplateSearch<1ul, NN_MAX_NUM_DIMENSIONS, detail::ForEach>::apply(first.dims(), func, first.shape(), first.strides(), first, ts...);
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

#ifndef TENSOR_H
#define TENSOR_H

#ifdef APPLE
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

#include <vector>
#include "op.h"
#include "error.h"

namespace nnlib
{

template <typename T>
class TensorBase
{
public:
	TensorBase(size_t n) : m_sizes(1, n), m_size(n), m_buffer(new T[m_size])
	{}
	
	/// Element access.
	T &operator[](size_t i)
	{
		return m_buffer[i];
	}
	
	size_t size() const
	{
		return m_size;
	}
protected:
	std::vector<size_t> m_sizes;
	size_t m_size;
	T *m_buffer;
};

/// Default Tensor.
template <typename T>
class Tensor : public TensorBase<T>
{
using TensorBase<T>::TensorBase;
};

/// Tensor specialization for double-precision floats.
template <>
class Tensor<double> : public TensorBase<double>
{
using TensorBase<double>::TensorBase;
public:
	/// Direct assignment.
	Tensor &operator=(const Tensor &t)
	{
		Assert(m_size == t.m_size, "Incompatible size for assignment!");
		cblas_dcopy(
			m_size,
			t.m_buffer, 1,
			m_buffer, 1
		);
		return *this;
	}
	
	/// Deferred evaluation of operations.
	template <typename T>
	Tensor &operator=(const OperatorAdd<Tensor, T> &op)
	{
		Assert(m_size == op.lhs.m_size, "Incompatible size for assignment!");
		*this = op.lhs;
		return *this += op.rhs;
	}
	
	/// Addition (deferred).
	template <typename T>
	OperatorAdd<Tensor, T> operator+(const T &other)
	{
		return OperatorAdd<Tensor, T>(*this, other);
	}
	
	/// Multiply (deferred).
	template <typename T>
	OperatorMultiply<Tensor, T> operator*(const T &other)
	{
		return OperatorMultiply<Tensor, T>(*this, other);
	}
	
	/// Addition.
	Tensor &operator+=(const Tensor &t)
	{
		Assert(m_size == t.m_size, "Incompatible sizes!");
		cblas_daxpy(
			m_size, 1,
			t.m_buffer, 1,
			m_buffer, 1
		);
		return *this;
	}
private:
	
};

}

#endif

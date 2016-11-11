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
#include "random.h"

namespace nnlib
{

template <typename T>
class TensorBase
{
public:
	TensorBase(size_t n) : m_sizes(1, n), m_size(n), m_buffer(new T[m_size])
	{}
	
	TensorBase(size_t rows, size_t cols) : m_sizes({ rows, cols }), m_size(rows * cols), m_buffer(new T[m_size])
	{}
	
	/// Element access.
	T &operator[](size_t i)
	{
		return m_buffer[i];
	}
	
	/// Number of elements in total.
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
	typedef double T;
	
	/// Fill this tensor using a normal distribution.
	void fillNormal(T mean = 0.0, T stddev = 1.0, T cap = 3.0)
	{
		Random r;
		for(size_t i = 0; i < m_size; ++i)
			m_buffer[i] = r.normal(mean, stddev, cap);
	}
	
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
	template <typename U>
	Tensor &operator=(const OperatorAdd<Tensor, U> &op)
	{
		Assert(m_size == op.lhs.m_size, "Incompatible size for assignment!");
		*this = op.lhs;
		return *this += op.rhs;
	}
	
	/// Addition (deferred).
	template <typename U>
	OperatorAdd<Tensor, U> operator+(const U &other)
	{
		return OperatorAdd<Tensor, U>(*this, other);
	}
	
	/// Multiply (deferred).
	template <typename U>
	OperatorMultiply<Tensor, U> operator*(const U &other)
	{
		return OperatorMultiply<Tensor, U>(*this, other);
	}
	
	/// Addition with another tensor.
	Tensor &operator+=(const Tensor &t)
	{
		Assert(m_size == t.m_size, "Incompatible sizes for addition!");
		cblas_daxpy(
			m_size, 1,
			t.m_buffer, 1,
			m_buffer, 1
		);
		return *this;
	}
	
	/// Addition with a dot product.
	Tensor &operator+=(const OperatorMultiply<Tensor, Tensor> &op)
	{
		Assert(m_size == op.lhs.m_sizes[0], "Incompatible sizes for dot product!");
		cblas_dgemv(
			CblasRowMajor,		// ordering
			CblasNoTrans,		// transpose
			op.lhs.m_sizes[0],	// rows
			op.lhs.m_sizes[1],	// cols
			1,					// scale of A
			op.lhs.m_buffer,	// A
			op.lhs.m_sizes[1],	// lda (length of continuous dimension)
			op.rhs.m_buffer,	// x
			1,					// stride of x
			1,					// scale of y
			m_buffer,			// y
			1					// stride of y
		);
		return *this;
	}
private:
	
};

}

#endif

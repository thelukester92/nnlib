#ifndef TENSOR_H
#define TENSOR_H

#ifdef APPLE
	#include <Accelerate/Accelerate.h>
#else
	#include <cblas.h>
#endif

#include <vector>
#include <type_traits>
#include "op.h"
#include "error.h"
#include "random.h"
#include "blas.h"

namespace nnlib
{

/// General n-dimensional tensor (with no specialized methods).
template <typename T>
class Tensor
{
public:
	/// General-purpose constructor.
	Tensor(size_t n) : m_size(n), m_capacity(n), m_buffer(new T[n])
	{}
	
	/// Reserve n elements in buffer.
	/// Elements in excess of m_size are unused.
	void reserve(size_t n)
	{
		if(n > m_capacity)
		{
			T *buffer = new T[m_capacity = n];
			for(size_t i = 0; i < m_size; ++i)
				buffer[i] = m_buffer[i];
			delete[] m_buffer;
			m_buffer = buffer;
		}
	}
	
	/// Set all elements to the given value.
	void fill(const T &val)
	{
		for(size_t i = 0; i < m_size; ++i)
			m_buffer[i] = val;
	}
	
	/// Set all elements to random values drawn from a normal distribution.
	/// This method only works if T is float or double.
	template <typename U = T>
	typename std::enable_if<std::is_same<U, float>::value || std::is_same<U, double>::value, void>::type fillNormal(Random &r, T mean = 0.0, T stddev = 1.0, T cap = 3.0)
	{
		for(size_t i = 0; i < m_size; ++i)
			m_buffer[i] = r.normal(mean, stddev, cap);
	}
	
	/// Raw element access.
	T &operator[](size_t i)
	{
		Assert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Raw element access.
	const T &operator[](size_t i) const
	{
		Assert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Number of elements in total.
	size_t size() const
	{
		return m_size;
	}
protected:
	size_t m_size, m_capacity;
	T *m_buffer;
};

template <typename T> class Vector;
template <typename T> class Matrix;

/// Matrices (2-dimensional tensors with matrix methods).
template <typename T>
class Matrix : public Tensor<T>
{
friend class Vector<T>;
using Tensor<T>::reserve;
using Tensor<T>::m_size;
using Tensor<T>::m_buffer;
public:
	/// General-purpose constructor.
	Matrix(size_t rows, size_t cols) : Tensor<T>(rows * cols), m_rows(rows), m_cols(cols)
	{}
	
	/// Change the dimensions of the matrix.
	void resize(size_t rows, size_t cols, const T &val = T())
	{
		reserve(rows * cols);
		size_t i = m_size;
		m_size = rows * cols;
		for(; i < m_size; ++i)
			m_buffer[i] = val;
	}
	
	/// Element access.
	T &operator()(size_t i, size_t j)
	{
		Assert(i < m_rows && j < m_cols, "Index out of bounds!");
		return m_buffer[i * m_cols + j];
	}
	
	/// Element access.
	const T &operator()(size_t i, size_t j) const
	{
		Assert(i < m_rows && j < m_cols, "Index out of bounds!");
		return m_buffer[i * m_cols + j];
	}
	
	/// Number of rows.
	size_t rows() const
	{
		return m_rows;
	}
	
	/// Number of columns.
	size_t cols() const
	{
		return m_cols;
	}
private:
	size_t m_rows, m_cols;
};

/// Vectors (1-dimensional tensors with vector methods).
template <typename T>
class Vector : public Tensor<T>
{
friend class Matrix<T>;
using Tensor<T>::reserve;
using Tensor<T>::m_size;
using Tensor<T>::m_buffer;
public:
	/// General-purpose constructor.
	Vector(size_t n) : Tensor<T>(n)
	{}
	
	/// Assign a vector.
	Vector &operator=(const Vector &v)
	{
		Assert(m_size == v.m_size, "Incompatible size!");
		BLAS<T>::copy(m_size, v.m_buffer, 1, m_buffer, 1);
		return *this;
	}
	
	/// Assign a sum.
	template <typename U, typename V>
	Vector &operator=(const OpAdd<U, V> &op)
	{
		*this = op.lhs;
		return *this += op.rhs;
	}
	
	/// Assign a product.
	Vector &operator=(const OpMult<Matrix<T>, Vector> &op)
	{
		Assert(m_size == op.lhs.m_rows && op.lhs.m_cols == op.rhs.m_size, "Incompatible multiplicands!");
		BLAS<T>::gemv(CblasRowMajor, CblasNoTrans, op.lhs.m_rows, op.lhs.m_cols, 1, op.lhs.m_buffer, op.lhs.m_cols, op.rhs.m_buffer, 1, 0, m_buffer, 1);
		return *this;
	}
	
	/// Change the dimensions of the vector.
	void resize(size_t n, const T &val = T())
	{
		reserve(n);
		size_t i = m_size;
		m_size = n;
		for(; i < m_size; ++i)
			m_buffer[i] = val;
	}
	
	/// Element access.
	T &operator()(size_t i)
	{
		Assert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Element access.
	const T &operator()(size_t i) const
	{
		Assert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Element-wise addition.
	Vector &operator+=(const Vector &v)
	{
		Assert(m_size == v.m_size, "Incompatible size!");
		BLAS<T>::axpy(m_size, 1, v.m_buffer, 1, m_buffer, 1);
		return *this;
	}
	
	/// Element-wise addition.
	template <typename U, typename V>
	Vector &operator+=(const OpAdd<U, V> &op)
	{
		*this += op.lhs;
		return *this += op.rhs;
	}
};

/// Deferred element-wise addition.
template <typename U, typename V>
OpAdd<U, V> operator+(const U &lhs, V &rhs)
{
	return OpAdd<U, V>(lhs, rhs);
}

/// Deferred tensor multiplication.
template <typename U, typename V>
OpMult<U, V> operator*(const U &lhs, const V &rhs)
{
	return OpMult<U, V>(lhs, rhs);
}

}

#endif

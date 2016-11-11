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

namespace nnlib
{

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
	typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value, void>::type fillNormal(Random &r, T mean = 0.0, T stddev = 1.0, T cap = 3.0)
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

template <typename T>
class Matrix : public Tensor<T>
{
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


/*
template <typename T>
class TensorBase
{
public:
	TensorBase() : m_sizes(2, 0), m_size(0), m_capacity(m_size), m_buffer(nullptr)
	{}
	
	TensorBase(size_t n) : m_sizes({ n, 1 }), m_size(n), m_capacity(m_size), m_buffer(new T[m_capacity])
	{}
	
	TensorBase(size_t rows, size_t cols) : m_sizes({ rows, cols }), m_size(rows * cols), m_capacity(m_size), m_buffer(new T[m_capacity])
	{}
	
	/// Change this into a 1-dimensional tensor of the given size and default value.
	void resize(size_t n)
	{
		reserve(n);
		m_sizes = { n, 1 };
		m_size = n;
	}
	
	/// Change this into a 2-dimensional vector of the given size and default value.
	void resize(size_t rows, size_t cols)
	{
		reserve(rows * cols);
		m_sizes = { rows, cols };
		m_size = rows * cols;
	}
	
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
	
	/// Element access (vector-style).
	T &operator[](size_t i)
	{
		Assert(i < m_size, "Index out-of-bounds!");
		return m_buffer[i];
	}
	
	/// Element access (vector-style).
	T &operator()(size_t i)
	{
		Assert(i < m_size, "Index out-of-bounds!");
		return m_buffer[i];
	}
	
	/// Element access (matrix-style).
	T &operator()(size_t i, size_t j)
	{
		Assert(i < m_sizes[0] && j < m_sizes[1], "Index out-of-bounds!");
		return m_buffer[i * m_sizes[1] + j];
	}
	
	/// Number of elements in total.
	size_t size() const
	{
		return m_size;
	}
protected:
	std::vector<size_t> m_sizes;
	size_t m_size, m_capacity;
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
	
	Tensor() {}
	
	/// Fill this tensor using a normal distribution.
	void fillNormal(Random &r, T mean = 0.0, T stddev = 1.0, T cap = 3.0)
	{
		for(size_t i = 0; i < m_size; ++i)
			m_buffer[i] = r.normal(mean, stddev, cap);
	}
	
	/// \todo matrix-matrix multiplication, don't assume matrix-vector
	/// \todo also, vector-vector dot product
	
	/// Construction from another tensor.
	Tensor(const Tensor &t) : TensorBase<T>(t.m_size)
	{
		cblas_dcopy(
			m_size,
			t.m_buffer, 1,
			m_buffer, 1
		);
	}
	
	/// Assignment to another tensor.
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
	
	/// Safe assignment to another tensor.
	void assignSafe(const Tensor &t)
	{
		resize(t.m_size);
		cblas_dcopy(
			m_size,
			t.m_buffer, 1,
			m_buffer, 1
		);
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
	
	/// Construction from a matrix-vector multiplication (evalulation of deferred multiplication).
	Tensor(const OperatorMultiply<Tensor, Tensor> &op) : TensorBase<T>(op.lhs.m_sizes[0])
	{
		cblas_dgemm(
			CblasRowMajor,		// ordering
			CblasNoTrans,		// transpose A
			CblasNoTrans,		// transpose B
			op.lhs.m_sizes[0],	// rows A and C
			op.rhs.m_sizes[1],	// cols B and C
			op.lhs.m_sizes[1],	// cols A and rows B
			1,					// scale of A and B
			op.lhs.m_buffer,	// A
			op.lhs.m_sizes[1],	// lda (length of continuous dimension A -- i.e. the y-stride)
			op.rhs.m_buffer,	// B
			op.rhs.m_sizes[1],	// ldb (length of continuous dimension B -- i.e. the y-stride)
			0,					// scale of C
			m_buffer,			// C
			m_sizes[1]			// ldc (length of continuous dimension C -- i.e. the y-stride)
		);
	}
	
	/// Assignment to a matrix-vector multiplication (evaluation of deferred multiplication).
	template <typename U>
	Tensor &operator=(const OperatorMultiply<Tensor, Tensor> &op)
	{
		Assert(m_size == op.lhs.m_sizes[0], "Incompatible sizes for dot product!");
		cblas_dgemm(
			CblasRowMajor,		// ordering
			CblasNoTrans,		// transpose A
			CblasNoTrans,		// transpose B
			op.lhs.m_sizes[0],	// rows A and C
			op.rhs.m_sizes[1],	// cols B and C
			op.lhs.m_sizes[1],	// cols A and rows B
			1,					// scale of A and B
			op.lhs.m_buffer,	// A
			op.lhs.m_sizes[1],	// lda (length of continuous dimension A -- i.e. the y-stride)
			op.rhs.m_buffer,	// B
			op.rhs.m_sizes[1],	// ldb (length of continuous dimension B -- i.e. the y-stride)
			0,					// scale of C
			m_buffer,			// C
			m_sizes[1]			// ldc (length of continuous dimension C -- i.e. the y-stride)
		);
		return *this;
	}
	
	/// Safe assignment to a matrix-vector multiplication (evaluation of deferred multiplication).
	void assignSafe(const OperatorMultiply<Tensor, Tensor> &op)
	{
		resize(op.lhs.m_sizes[0]);
		cblas_dgemm(
			CblasRowMajor,		// ordering
			CblasNoTrans,		// transpose A
			CblasNoTrans,		// transpose B
			op.lhs.m_sizes[0],	// rows A and C
			op.rhs.m_sizes[1],	// cols B and C
			op.lhs.m_sizes[1],	// cols A and rows B
			1,					// scale of A and B
			op.lhs.m_buffer,	// A
			op.lhs.m_sizes[1],	// lda (length of continuous dimension A -- i.e. the y-stride)
			op.rhs.m_buffer,	// B
			op.rhs.m_sizes[1],	// ldb (length of continuous dimension B -- i.e. the y-stride)
			0,					// scale of C
			m_buffer,			// C
			m_sizes[1]			// ldc (length of continuous dimension C -- i.e. the y-stride)
		);
	}
	
	/// Addition with a matrix-vector multiplication (evaluation of deferred multiplication).
	Tensor &operator+=(const OperatorMultiply<Tensor, Tensor> &op)
	{
		Assert(m_size == op.lhs.m_sizes[0], "Incompatible sizes for dot product!");
		cblas_dgemm(
			CblasRowMajor,		// ordering
			CblasNoTrans,		// transpose A
			CblasNoTrans,		// transpose B
			op.lhs.m_sizes[0],	// rows A and C
			op.rhs.m_sizes[1],	// cols B and C
			op.lhs.m_sizes[1],	// cols A and rows B
			1,					// scale of A and B
			op.lhs.m_buffer,	// A
			op.lhs.m_sizes[1],	// lda (length of continuous dimension A -- i.e. the y-stride)
			op.rhs.m_buffer,	// B
			op.rhs.m_sizes[1],	// ldb (length of continuous dimension B -- i.e. the y-stride)
			1,					// scale of C
			m_buffer,			// C
			m_sizes[1]			// ldc (length of continuous dimension C -- i.e. the y-stride)
		);
		return *this;
	}
	
	/// Construction from a sum (evalulation of deferred addition).
	template <typename U, typename V>
	Tensor(const OperatorAdd<U, V> &op)
	{
		assignSafe(op);
	}
	
	/// Assignment to a sum (evaluation of deferred addition).
	template <typename U, typename V>
	Tensor &operator=(const OperatorAdd<U, V> &op)
	{
		*this = op.lhs;
		return *this += op.rhs;
	}
	
	/// Safe assignment to a sum (evaluation of deferred addition).
	template <typename U, typename V>
	void assignSafe(const OperatorAdd<U, V> &op)
	{
		assignSafe(op.lhs);
		*this += op.rhs;
	}
	
	/// Addition with a sum (i.e. more than two addends; evaluation of deferred addition).
	template <typename U, typename V>
	Tensor &operator+=(const OperatorAdd<U, V> &op)
	{
		*this += op.lhs;
		return *this += op.rhs;
	}
	
	/// Addition (deferred).
	template <typename U>
	OperatorAdd<Tensor, U> operator+(const U &other)
	{
		return OperatorAdd<Tensor, U>(*this, other);
	}
	
	/// Multiplication (deferred).
	template <typename U>
	OperatorMultiply<Tensor, U> operator*(const U &other)
	{
		return OperatorMultiply<Tensor, U>(*this, other);
	}
};

/// Addition (deferred).
template <typename U, typename V>
OperatorAdd<U, V> operator+(const U &lhs, const V &rhs)
{
	return OperatorAdd<U, V>(lhs, rhs);
}

/// Multiplication (deferred).
template <typename U, typename V>
OperatorMultiply<U, V> operator*(const U &lhs, const V &rhs)
{
	return OperatorMultiply<U, V>(lhs, rhs);
}

*/

}

#endif

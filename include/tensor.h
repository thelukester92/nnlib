#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <type_traits>
#include "op.h"
#include "error.h"
#include "random.h"
#include "blas.h"

namespace nnlib
{

/// Tensor base class (with no specialized methods).
template <typename T>
class Tensor
{
public:
	/// General-purpose constructor.
	Tensor(size_t n)
	: m_size(n), m_capacity(n), m_buffer(new T[n]), m_sharedBuffer(m_buffer), m_sharedSize(0)
	{}
	
	/// Create a shared-memory tensor.
	Tensor(std::shared_ptr<T> buffer, size_t bufSize, size_t n, size_t offset = 0)
	: m_size(n), m_capacity(n), m_buffer(&*buffer + offset), m_sharedBuffer(buffer), m_sharedSize(bufSize)
	{}
	
	/// Create a new shared-memory tensor from another Tensor.
	Tensor(Tensor &t, size_t n, size_t offset = 0)
	: m_buffer(t.m_buffer + offset), m_size(n), m_capacity(n), m_sharedBuffer(t.m_sharedBuffer), m_sharedSize(t.m_sharedSize)
	{}
	
	/// Assign a new shared-memory buffer.
	void shareBuffer(std::shared_ptr<T> buffer, size_t bufSize, size_t n, size_t offset = 0)
	{
		m_buffer = &*buffer + offset;
		m_size = n;
		m_capacity = n;
		m_sharedBuffer = buffer;
		m_sharedSize = bufSize;
	}
	
	/// Assign a new shared-memory buffer from another Tensor.
	void shareBuffer(Tensor &t, size_t n, size_t offset = 0)
	{
		m_buffer = t.m_buffer + offset;
		m_size = n;
		m_capacity = n;
		m_sharedBuffer = t.m_sharedBuffer;
		m_sharedSize = t.m_sharedSize;
	}
	
	/// Set a new data offset from the actual buffer.
	void offsetBuffer(size_t n)
	{
		Assert(n < m_sharedSize, "Cannot offset beyond the end of the Tensor storage!");
		m_buffer = &*m_sharedBuffer + n;
		m_size = m_sharedSize - n;
		m_capacity = m_sharedSize - n;
	}
	
	/// Reserve n elements in buffer.
	/// Elements in excess of m_size are unused.
	/// This is not allowed for a shared Tensor.
	void reserve(size_t n)
	{
		if(n > m_capacity)
		{
			Assert(m_sharedSize == 0, "Cannot reserve more space in a shared Tensor!");
			T *buffer = new T[m_capacity = n];
			for(size_t i = 0; i < m_size; ++i)
				buffer[i] = m_buffer[i];
			m_buffer = buffer;
			m_sharedBuffer.reset(m_buffer); // this will delete the old buffer
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
	size_t m_size, m_capacity;			///< number of elements and size of the buffer (minus offset)
	T *m_buffer;						///< pointer to the buffer (with offset).
	std::shared_ptr<T> m_sharedBuffer;	///< the original allocated buffer; this handles deleting m_buffer.
	size_t m_sharedSize;				///< the size of the shared buffer, if shared, or 0 if not shared.
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
	
	/// Assign a product.
	/// \todo verify the math here, I did this in a rush.
	Matrix &operator=(const OpMult<Vector<T>, Vector<T>> &op)
	{
		Assert(m_rows == op.lhs.m_size && m_cols == op.rhs.m_size, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_rows, m_cols, 1, 1, op.lhs.m_buffer, 1, op.rhs.m_buffer, op.rhs.m_size, 0, m_buffer, m_cols);
		return *this;
	}
	
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
	/// Create a new vector that manages data for the given Tensors.
	static Vector flatten(const std::vector<Tensor<T> *> &tensors)
	{
		size_t n = 0;
		for(auto t : tensors)
			n += t->size();
		
		Vector v(n);
		
		
		return v;
	}
	
	/// General-purpose constructor.
	Vector(size_t n) : Tensor<T>(n)
	{}
	
	/// Construct from a vector.
	Vector(const Vector &v) : Tensor<T>(v.m_size)
	{
		BLAS<T>::copy(m_size, v.m_buffer, 1, m_buffer, 1);
	}
	
	/// Construct from a sum.
	template <typename U, typename V>
	Vector(const OpAdd<U, V> &op) : Tensor<T>(0)
	{
		safeAssign(op.lhs);
		*this += op.rhs;
	}
	
	/// Construct from a product.
	Vector(const OpMult<Matrix<T>, Vector> &op) : Tensor<T>(op.lhs.m_rows)
	{
		Assert(op.lhs.m_cols == op.rhs.m_size, "Incompatible multiplicands!");
		BLAS<T>::gemv(CblasRowMajor, CblasNoTrans, op.lhs.m_rows, op.lhs.m_cols, 1, op.lhs.m_buffer, op.lhs.m_cols, op.rhs.m_buffer, 1, 0, m_buffer, 1);
	}
	
	/// Construct from a product.
	Vector(const OpMult<Vector, Matrix<T>> &op) : Tensor<T>(op.rhs.m_rows)
	{
		Assert(op.lhs.m_size == op.rhs.m_rows, "Incompatible multiplicands!");
		BLAS<T>::gemv(CblasRowMajor, CblasTrans, op.rhs.m_rows, op.rhs.m_cols, 1, op.rhs.m_buffer, op.rhs.m_cols, op.lhs.m_buffer, 1, 0, m_buffer, 1);
	}
	
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
	
	/// Assign a product.
	Vector &operator=(const OpMult<Vector, Matrix<T>> &op)
	{
		Assert(op.lhs.m_size == op.rhs.m_rows, "Incompatible multiplicands!");
		BLAS<T>::gemv(CblasRowMajor, CblasTrans, op.rhs.m_rows, op.rhs.m_cols, 1, op.rhs.m_buffer, op.rhs.m_cols, op.lhs.m_buffer, 1, 0, m_buffer, 1);
		return *this;
	}
	
	/// Assign-and-resize a vector.
	void safeAssign(const Vector &v)
	{
		resize(v.m_size);
		BLAS<T>::copy(m_size, v.m_buffer, 1, m_buffer, 1);
	}
	
	/// Assign-and-resize a sum.
	template <typename U, typename V>
	void safeAssign(const OpAdd<U, V> &op)
	{
		safeAssign(op.lhs);
		*this += op.rhs;
	}
	
	/// Assign-and-resize a product.
	void safeAssign(const OpMult<Matrix<T>, Vector> &op)
	{
		resize(op.lhs.m_rows);
		Assert(op.lhs.m_cols == op.rhs.m_size, "Incompatible multiplicands!");
		BLAS<T>::gemv(CblasRowMajor, CblasNoTrans, op.lhs.m_rows, op.lhs.m_cols, 1, op.lhs.m_buffer, op.lhs.m_cols, op.rhs.m_buffer, 1, 0, m_buffer, 1);
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
OpAdd<U, V> operator+(const U &lhs, const V &rhs)
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

#ifndef TENSOR_H
#define TENSOR_H

#include <type_traits>
#include "op.h"
#include "error.h"
#include "blas.h"

namespace nnlib
{

/// An ordered collection of elements of type T.
/// Abstract base class for other tensors.
template <typename T>
class Tensor
{
public:
	/// General-purpose constructor.
	Tensor(size_t n, const T &val = T())
	: m_size(n), m_capacity(n), m_buffer(new T[m_capacity]), m_sharedSize(n), m_sharedBuffer(m_buffer)
	{
		BLAS<T>::set(n, val, m_buffer, 1);
	}
	
	/// Shared-data constructor.
	Tensor(Tensor &t, size_t n, size_t offset = 0)
	: m_size(n), m_capacity(n), m_buffer(t.m_buffer + offset), m_sharedSize(t.m_sharedSize), m_sharedBuffer(t.m_sharedBuffer)
	{}
	
	/// Destructor; pure virtual to make this an abstract class.
	virtual ~Tensor() = 0;
	
	/// Reserve more space in this tensor.
	void reserve(size_t n)
	{
		if(n > m_capacity)
		{
			T *buffer = new T[m_capacity = n];
			for(size_t i = 0; i < m_size; ++i)
				buffer[i] = m_buffer[i];
			m_buffer = buffer;
			m_sharedBuffer.reset(m_buffer); // this will delete the old buffer
		}
	}
	
	/// The number of elements in this tensor.
	size_t size() const
	{
		return m_size;
	}
protected:
	size_t m_size, m_capacity;
	T *m_buffer;
	size_t m_sharedSize;
	std::shared_ptr<T> m_sharedBuffer;
};

template <typename T>
inline Tensor<T>::~Tensor() {}

/// A 1-dimensional tensor; may not be dense.
template <typename T>
class Vector : public Tensor<T>
{
using Tensor<T>::m_size;
using Tensor<T>::m_buffer;
using Tensor<T>::m_sharedBuffer;
using Tensor<T>::m_sharedSize;
public:
	/// General purpose constructor.
	Vector(size_t size = 0, const T &val = T())
	: Tensor<T>(size, val), m_stride(1)
	{}
	
	/// Copy constructor.
	Vector(const Vector &v)
	: Tensor<T>(v.m_size), m_stride(1)
	{
		copy(v);
	}
	
	/// Shared-data constructor.
	Vector(Vector<T> &v, size_t n, size_t offset = 0)
	: Tensor<T>(v, n, offset), m_stride(v.m_stride)
	{}
	
	/// Shared-data constructor.
	Vector(Tensor<T> &t, size_t n, size_t offset = 0)
	: Tensor<T>(t, n, offset), m_stride(1)
	{}
	
	/// Copy assignment.
	Vector &operator=(const Vector &v)
	{
		copy(v);
		return *this;
	}
	
	/// Copy the elements from v into this.
	void copy(const Vector &v)
	{
		NNAssert(v.m_size == m_size, "Invalid size!");
		BLAS<T>::copy(m_size, v.m_buffer, v.m_stride, m_buffer, m_stride);
	}
	
	/// Set the offset of this vector.
	void setOffset(size_t n)
	{
		NNAssert(n + m_size <= m_sharedSize, "Invalid offset!");
		m_buffer = m_sharedBuffer.get() + n;
	}
	
	/// Fill this tensor with the given value.
	void fill(const T &val)
	{
		BLAS<T>::set(m_size, val, m_buffer, m_stride);
	}
	
	/// Element access.
	T &operator[](size_t i)
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i * m_stride];
	}
	
	/// Element access.
	const T &operator[](size_t i) const
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i * m_stride];
	}
	
	/// Element access.
	T &at(size_t i)
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i * m_stride];
	}
	
	/// Element access.
	const T &at(size_t i) const
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i * m_stride];
	}
	
	/// Element access.
	T &operator()(size_t i)
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i * m_stride];
	}
	
	/// Element access.
	const T &operator()(size_t i) const
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i * m_stride];
	}
	
	/// Element-wise addition.
	void add(const Vector &v)
	{
		NNAssert(m_size == v.m_size, "Incompatible addends!");
		BLAS<T>::axpy(m_size, 1, v.m_buffer, v.m_stride, m_buffer, m_stride);
	}
	
	/// Element-wise addition.
	Vector &operator+=(const Vector &v)
	{
		NNAssert(m_size == v.m_size, "Incompatible addends!");
		BLAS<T>::axpy(m_size, 1, v.m_buffer, v.m_stride, m_buffer, m_stride);
		return *this;
	}
	
	/// Element-wise scaling.
	void scale(const T &scalar)
	{
		BLAS<T>::scal(m_size, scalar, m_buffer, m_stride);
	}
	
	/// Element-wise scaling.
	Vector &operator*=(const T &scalar)
	{
		BLAS<T>::scal(m_size, scalar, m_buffer, m_stride);
		return *this;
	}

private:
	size_t m_stride;
};

/// A 2-dimensional tensor.
/// Adjacent columns are contiguous, but rows may be separated by more than cols.
template <typename T>
class Matrix : public Tensor<T>
{
using Tensor<T>::m_size;
using Tensor<T>::m_buffer;
using Tensor<T>::reserve;
public:
	/// General-purpose constructor.
	Matrix(size_t rows, size_t cols)
	: Tensor<T>(rows * cols), m_rows(rows), m_cols(cols), m_ld(cols)
	{}
	
	/// Copy constructor.
	Matrix(const Matrix &m)
	: Tensor<T>(m.m_size), m_rows(m.m_rows), m_cols(m.m_cols), m_ld(m.m_cols)
	{
		copy(m);
	}
	
	/// Shared-data constructor.
	Matrix(Matrix &m, size_t rows, size_t cols, size_t rowOffset = 0, size_t colOffset = 0)
	: Tensor<T>(m, rows * cols, m.m_ld * rowOffset + colOffset), m_rows(rows), m_cols(cols), m_ld(m.m_ld)
	{}
	
	/// Operation constructor.
	Matrix(const Operation<Matrix> &op)
	{
		op.assign(*this);
	}
	
	/// Operation assignment.
	Matrix &operator=(const Operation<Matrix> &op)
	{
		op.assign(*this);
		return *this;
	}
	
	/// Copy assignment.
	Matrix &operator=(const Matrix &m)
	{
		copy(m);
		return *this;
	}
	
	/// Resize this matrix.
	void resize(size_t rows, size_t cols)
	{
		reserve(rows * cols);
		if(m_ld == m_cols)
			m_ld = cols;
		m_rows = rows;
		m_cols = cols;
	}
	
	/// Copy the elements from m into this.
	void copy(const Matrix &m)
	{
		resize(m.m_rows, m.m_cols);
		T *from = m.m_buffer, *to = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, from += m.m_ld, to += m_ld)
			BLAS<T>::copy(m_cols, from, 1, to, 1);
	}
	
	/// Fill this tensor with the given value.
	void fill(const T &val)
	{
		T *buffer = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, buffer += m_ld)
			BLAS<T>::set(m_cols, val, buffer, 1);
	}
	
	/// Element access.
	T &at(size_t i, size_t j)
	{
		NNAssert(i < m_rows && j < m_cols, "Index out of bounds!");
		return m_buffer[i * m_ld + j];
	}
	
	/// Element access.
	const T &at(size_t i, size_t j) const
	{
		NNAssert(i < m_rows && j < m_cols, "Index out of bounds!");
		return m_buffer[i * m_ld + j];
	}
	
	/// Element access.
	T &operator()(size_t i, size_t j)
	{
		NNAssert(i < m_rows && j < m_cols, "Index out of bounds!");
		return m_buffer[i * m_ld + j];
	}
	
	/// Element access.
	const T &operator()(size_t i, size_t j) const
	{
		NNAssert(i < m_rows && j < m_cols, "Index out of bounds!");
		return m_buffer[i * m_ld + j];
	}
	
	/// Element access.
	T &operator[](size_t i)
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Element access.
	const T &operator[](size_t i) const
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Element-wise addition.
	void add(const Matrix &m)
	{
		NNAssert(m_rows == m.m_rows && m_cols == m.m_cols, "Incompatible addends!");
		T *from = m.m_buffer, *to = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, from += m.m_ld, to += m_ld)
			BLAS<T>::axpy(m_cols, 1, from, 1, to, 1);
	}
	
	/// Element-wise addition.
	Matrix &operator+=(const Matrix &m)
	{
		NNAssert(m_rows == m.m_rows && m_cols == m.m_cols, "Incompatible addends!");
		T *from = m.m_buffer, *to = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, from += m.m_ld, to += m_ld)
			BLAS<T>::axpy(m_cols, 1, from, 1, to, 1);
		return *this;
	}
	
	/// Element-wise scaling.
	void scale(const T &scalar)
	{
		T *buffer = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, buffer += m_ld)
			BLAS<T>::scal(m_cols, scalar, buffer, 1);
	}
	
	/// Element-wise scaling.
	Matrix &operator*=(const T &scalar)
	{
		T *buffer = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, buffer += m_ld)
			BLAS<T>::scal(m_cols, scalar, buffer, 1);
		return *this;
	}
	
private:
	size_t m_rows, m_cols, m_ld;
};

/// Addition operator overload.
template <typename T>
OperationAdd<T> operator+(const T &lhs, const T &rhs)
{
	return OperationAdd<T>(lhs, rhs);
}










/*
/// \todo give operations a size for the result to make safeAssign only need one version
/// \todo use lda/ldb/ldc instead of m_cols, since m_cols may be less if looking at a shared part of the matrix
/// \todo Vector = Matrix with one row

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
	: m_size(n), m_capacity(n), m_buffer(t.m_buffer + offset), m_sharedBuffer(t.m_sharedBuffer), m_sharedSize(t.m_sharedSize)
	{}
	
	/// Assign a new shared-memory buffer.
	void shareBuffer(std::shared_ptr<T> buffer, size_t bufSize, size_t n, size_t offset = 0)
	{
		m_buffer = buffer.get() + offset;
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
	
	/// Erase element i from this tensor.
	void erase(size_t i)
	{
		Assert(i < m_size, "Invalid erase index!");
		for(; i < m_size - 1; ++i)
			m_buffer[i] = m_buffer[i + 1];
		--m_size;
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
	
	/// Raw element access.
	T &at(size_t i)
	{
		Assert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Raw element access.
	const T &at(size_t i) const
	{
		Assert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Get the last element. There must be at least one element in this tensor.
	T &back()
	{
		Assert(m_size > 0, "Cannot get the back element of an empty tensor!");
		return m_buffer[m_size - 1];
	}
	
	/// Get the last element. There must be at least one element in this tensor.
	const T &back() const
	{
		Assert(m_size > 0, "Cannot get the back element of an empty tensor!");
		return m_buffer[m_size - 1];
	}
	
	/// Insert an element at the end, resizing if necessary.
	void push_back(const T &val)
	{
		reserve(m_size + 1);
		m_buffer[m_size++] = val;
	}
	
	/// Number of elements in total.
	size_t size() const
	{
		return m_size;
	}
	
	/// std-like iterator begin.
	T *begin()
	{
		return m_buffer;
	}
	
	/// std-like iterator begin.
	const T *begin() const
	{
		return m_buffer;
	}
	
	/// std-like iterator end.
	T *end()
	{
		return m_buffer + m_size;
	}
	
	/// std-like iterator end.
	const T *end() const
	{
		return m_buffer + m_size;
	}
	
	/// Raw buffer.
	T *buffer()
	{
		return m_buffer;
	}
	
	/// Raw buffer.
	const T *buffer() const
	{
		return m_buffer;
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
	
	/// Construct a shared-data matrix.
	Matrix(Tensor<T> &t, size_t rows, size_t cols, size_t offset = 0) : Tensor<T>(t, rows * cols, offset), m_rows(rows), m_cols(cols)
	{}
	
	/// Construct from an operation.
	Matrix(const Op &op) : Tensor<T>(0), m_rows(0), m_cols(0)
	{
		safeAssign(op);
	}
	
	/// Assign a product.
	Matrix &operator=(const OpMult<Matrix, Matrix> &op)
	{
		Assert(m_rows == op.lhs.m_rows && m_cols == op.rhs.m_cols && op.lhs.m_cols == op.rhs.m_rows, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m_rows, m_cols, op.lhs.m_cols, 1, op.lhs.m_buffer, op.lhs.m_cols, op.rhs.m_buffer, op.rhs.m_cols, 0, m_buffer, m_cols);
		return *this;
	}
	
	/// Assign a product (with transposition on LHS).
	Matrix &operator=(const OpMult<OpTrans<Matrix>, Matrix> &op)
	{
		Assert(m_rows == op.lhs.target.m_cols && m_cols == op.rhs.m_cols && op.lhs.target.m_rows == op.rhs.m_rows, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasTrans, CblasNoTrans, m_rows, m_cols, op.lhs.target.m_rows, 1, op.lhs.target.m_buffer, op.lhs.target.m_cols, op.rhs.m_buffer, op.rhs.m_cols, 0, m_buffer, m_cols);
		return *this;
	}
	
	/// Assign a product (with transposition on RHS).
	Matrix &operator=(const OpMult<Matrix, OpTrans<Matrix>> &op)
	{
		Assert(m_rows == op.lhs.m_rows && m_cols == op.rhs.target.m_rows && op.lhs.m_cols == op.rhs.target.m_cols, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasNoTrans, CblasTrans, m_rows, m_cols, op.lhs.m_cols, 1, op.lhs.m_buffer, op.lhs.m_cols, op.rhs.target.m_buffer, op.rhs.target.m_cols, 0, m_buffer, m_cols);
		return *this;
	}
	
	/// Assign a product (with transposition on LHS and RHS).
	Matrix &operator=(const OpMult<OpTrans<Matrix>, OpTrans<Matrix>> &op)
	{
		Assert(m_rows == op.lhs.target.m_cols && m_cols == op.rhs.target.m_rows && op.lhs.target.m_rows == op.rhs.target.m_cols, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasTrans, CblasTrans, m_rows, m_cols, op.lhs.target.m_rows, 1, op.lhs.target.m_buffer, op.lhs.target.m_cols, op.rhs.target.m_buffer, op.rhs.target.m_cols, 0, m_buffer, m_cols);
		return *this;
	}
	
	/// Assign a sum.
	template <typename U, typename V>
	Matrix &operator=(const OpAdd<U, V> &op)
	{
		*this = op.lhs;
		return *this += op.rhs;
	}
	
	/// Assign-and-resize a product.
	void safeAssign(const OpMult<Matrix, Matrix> &op)
	{
		resize(op.lhs.m_rows, op.rhs.m_cols);
		*this = op;
	}
	
	/// Assign-and-resize a product (with transposition on LHS).
	void safeAssign(const OpMult<OpTrans<Matrix>, Matrix> &op)
	{
		resize(op.lhs.target.m_cols, op.rhs.m_cols);
		*this = op;
	}
	
	/// Assign-and-resize a product (with transposition on RHS).
	void safeAssign(const OpMult<Matrix, OpTrans<Matrix>> &op)
	{
		resize(op.lhs.m_rows, op.rhs.target.m_rows);
		*this = op;
	}
	
	/// Assign-and-resize a product (with transposition on LHS and RHS).
	void safeAssign(const OpMult<OpTrans<Matrix>, OpTrans<Matrix>> &op)
	{
		resize(op.lhs.target.m_cols, op.rhs.target.m_rows);
		*this = op;
	}
	
	/// Assign-and-resize a sum.
	template <typename U, typename V>
	void safeAssign(const OpAdd<U, V> &op)
	{
		safeAssign(op.lhs);
		*this += op.rhs;
	}
	
	/// Assign a new shared-memory buffer from another Tensor.
	void share(Matrix &m)
	{
		m_buffer = m.m_buffer;
		m_size = m.m_size;
		m_capacity = m.m_capacity;
		m_sharedBuffer = m.m_sharedBuffer;
		m_sharedSize = m.m_sharedSize;
		m_rows = m.m_rows;
		m_cols = m.m_cols;
	}
	
	/// Change the dimensions of the matrix.
	void resize(size_t rows, size_t cols, const T &val = T())
	{
		reserve(rows * cols);
		size_t i = m_size;
		m_size = rows * cols;
		m_rows = rows;
		m_cols = cols;
		for(; i < m_size; ++i)
			m_buffer[i] = val;
	}
	
	/// Vector access.
	/// \todo add column access.
	Vector<T> row(size_t i)
	{
		return Vector<T>(*this, m_cols, m_cols * i);
	}
	
	/// Vector access.
	void row(size_t i, Vector<T> &r)
	{
		r.shareBuffer(*this, m_cols, m_cols * i);
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
	
	/// Element-wise addition.
	Matrix &operator+=(const Matrix &m)
	{
		Assert(m_rows == m.m_rows && m_cols == m.m_cols, "Incompatible size!");
		BLAS<T>::axpy(m_size, 1, m.m_buffer, 1, m_buffer, 1);
		return *this;
	}
	
	/// Element-wise addition (repeating v for each row).
	Matrix &operator+=(const Vector<T> &v)
	{
		Assert(m_cols == v.m_size, "Incompatible size!");
		T *buffer = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, buffer += m_cols)
			BLAS<T>::axpy(m_cols, 1, v.m_buffer, 1, buffer, 1);
		return *this;
	}
	
	/// Element-wise addition.
	template <typename U, typename V>
	Matrix &operator+=(const OpAdd<U, V> &op)
	{
		*this += op.lhs;
		return *this += op.rhs;
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
	static Vector flatten(const Vector<Tensor<T> *> &tensors)
	{
		size_t n = 0, offset = 0;
		for(auto t : tensors)
			n += t->size();
		
		Vector v(n);
		for(auto t : tensors)
		{
			/// copy data from old buffer
			for(size_t i = 0; i < t->size(); ++i)
				v[offset + i] = t->at(i);
			
			/// assign new buffer
			t->shareBuffer(v, t->size(), offset);
			offset += t->size();
		}
		
		return v;
	}
	
	/// General-purpose constructor.
	explicit Vector(size_t n = 0) : Tensor<T>(n)
	{}
	
	/// Construct from a vector.
	Vector(const Vector &v) : Tensor<T>(v.m_size)
	{
		BLAS<T>::copy(m_size, v.m_buffer, 1, m_buffer, 1);
	}
	
	/// Construct from an initializer list.
	Vector(const std::initializer_list<T> &list) : Tensor<T>(list.size())
	{
		size_t i = 0;
		for(auto &val : list)
			m_buffer[i++] = val;
	}
	
	/// Construct a shared-memory vector from another Tensor.
	Vector(Tensor<T> &t, size_t n, size_t offset = 0) : Tensor<T>(t, n, offset)
	{}
	
	/// Construct from a sum.
	template <typename U, typename V>
	Vector(const OpAdd<U, V> &op) : Tensor<T>(0)
	{
		safeAssign(op.lhs);
		*this += op.rhs;
	}
	
	/// Construct from a difference.
	template <typename U, typename V>
	Vector(const OpSub<U, V> &op) : Tensor<T>(0)
	{
		safeAssign(op.lhs);
		*this -= op.rhs;
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
	
	/// Assign a difference.
	template <typename U, typename V>
	Vector &operator=(const OpSub<U, V> &op)
	{
		*this = op.lhs;
		return *this -= op.rhs;
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
	
	/// Assign a collapsed matrix.
	Vector &operator=(const OpCollapse<Matrix<T>> &op)
	{
		if(op.collapseCols)
		{
			Assert(m_size == op.target.m_rows, "Incompatible size!");
			T *buffer = op.target.m_buffer;
			BLAS<T>::copy(m_size, buffer, op.target.m_cols, m_buffer, 1);
			buffer += op.target.m_rows;
			for(size_t i = 1; i < op.target.m_cols; ++i, buffer += op.target.m_rows)
				BLAS<T>::axpy(m_size, 1, buffer, op.target.m_cols, m_buffer, 1);
		}
		else
		{
			Assert(m_size == op.target.m_cols, "Incompatible size!");
			T *buffer = op.target.m_buffer;
			BLAS<T>::copy(m_size, buffer, 1, m_buffer, 1);
			buffer += op.target.m_cols;
			for(size_t i = 1; i < op.target.m_rows; ++i, buffer += op.target.m_cols)
				BLAS<T>::axpy(m_size, 1, buffer, 1, m_buffer, 1);
		}
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
	
	/// Element-wise subtraction.
	Vector &operator-=(const Vector &v)
	{
		Assert(m_size == v.m_size, "Incompatible size!");
		BLAS<T>::axpy(m_size, -1, v.m_buffer, 1, m_buffer, 1);
		return *this;
	}
	
	/// Element-wise subtraction.
	template <typename U, typename V>
	Vector &operator-=(const OpAdd<U, V> &op)
	{
		*this -= op.lhs;
		return *this -= op.rhs;
	}
};

/// Deferred element-wise addition.
template <typename U, typename V>
OpAdd<U, V> operator+(const U &lhs, const V &rhs)
{
	return OpAdd<U, V>(lhs, rhs);
}

/// Deferred element-wise subtraction.
template <typename U, typename V>
OpSub<U, V> operator-(const U &lhs, const V &rhs)
{
	return OpSub<U, V>(lhs, rhs);
}

/// Deferred tensor multiplication.
template <typename U, typename V>
OpMult<U, V> operator*(const U &lhs, const V &rhs)
{
	return OpMult<U, V>(lhs, rhs);
}

/// Deferred matrix transposition.
template <typename U>
OpTrans<U> operator~(const U &target)
{
	return OpTrans<U>(target);
}
*/

}

#endif

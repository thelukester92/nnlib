#ifndef TENSOR_H
#define TENSOR_H

#include "operation.h"
#include "error.h"
#include "blas.h"

namespace nnlib
{

/// \todo better tensor iterator that is not undefined for non-contiguous tensors.
///       it should at least cause a runtime error.

/// An ordered collection of elements of type T.
/// Abstract base class for other tensors.
template <typename T>
class Tensor
{
public:
	/// General-purpose constructor.
	/// \todo handle zero-length as a special case.
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
	
	/// Reassign to use shared data.
	void shareBuffer(Tensor &t, size_t n, size_t offset = 0)
	{
		m_size = m_capacity = n;
		m_buffer = t.m_buffer + offset;
		m_sharedSize = t.m_sharedSize;
		m_sharedBuffer = t.m_sharedBuffer;
	}
	
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
	
	/// Actually change the number of elements in this tensor.
	void resize(size_t n, const T &val = T())
	{
		reserve(n);
		if(int(n) - int(m_size) > 0)
			BLAS<T>::set(n - m_size, val, m_buffer + m_size, 1);
		m_size = n;
	}
	
	/// The number of elements in this tensor.
	size_t size() const
	{
		return m_size;
	}
	
	/// Element access.
	virtual T &operator[](size_t i)
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// Element access.
	virtual const T &operator[](size_t i) const
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i];
	}
	
	/// std-like iterator begin.
	/// Behavior is undefined for non-contiguous tensors.
	T *begin()
	{
		return m_buffer;
	}
	
	/// std-like iterator end.
	/// Behavior is undefined for non-contiguous tensors.
	T *end()
	{
		return m_buffer + m_size;
	}
	
	/// std-like iterator begin.
	/// Behavior is undefined for non-contiguous tensors.
	const T *begin() const
	{
		return m_buffer;
	}
	
	/// std-like iterator end.
	/// Behavior is undefined for non-contiguous tensors.
	const T *end() const
	{
		return m_buffer + m_size;
	}
protected:
	size_t m_size, m_capacity;
	T *m_buffer;
	size_t m_sharedSize;
	std::shared_ptr<T> m_sharedBuffer;
};

template <typename T>
inline Tensor<T>::~Tensor() {}

template <typename T>
class Matrix;

/// A 1-dimensional tensor; may not be dense.
template <typename T>
class Vector : public Tensor<T>
{
friend class Matrix<T>;
using Tensor<T>::m_size;
using Tensor<T>::m_buffer;
using Tensor<T>::m_sharedBuffer;
using Tensor<T>::m_sharedSize;
public:
	using Tensor<T>::resize;
	
	/// Combine the given tensors into a single vector, then make the given tensors share data.
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
				v[offset + i] = (*t)[i];
			
			/// assign new buffer
			t->shareBuffer(v, t->size(), offset);
			offset += t->size();
		}
		
		return v;
	}
	
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
	
	/// Initializer list constructor.
	Vector(const std::initializer_list<T> &list)
	: Tensor<T>(list.size()), m_stride(1)
	{
		size_t i = 0;
		for(auto &val : list)
			m_buffer[i++] = val;
	}
	
	/// Operation constructor.
	Vector(const Operation<Vector> &op)
	: Tensor<T>(0), m_stride(1)
	{
		op.assign(*this);
	}
	
	/// Operation assignment.
	Vector &operator=(const Operation<Vector> &op)
	{
		op.assign(*this);
		return *this;
	}
	
	/// Copy assignment.
	Vector &operator=(const Vector &v)
	{
		resize(v.m_size);
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
	
	/// Add a single element.
	void push_back(const T &val)
	{
		resize(m_size + 1, val);
	}
	
	/// Remove a single element.
	void erase(size_t i)
	{
		NNAssert(i < m_size, "Invalid erase index!");
		for(; i < m_size - 1; ++i)
			m_buffer[i * m_stride] = m_buffer[(i + 1) * m_stride];
		--m_size;
	}
	
	/// Element access.
	virtual T &operator[](size_t i) override
	{
		NNAssert(i < m_size, "Index out of bounds!");
		return m_buffer[i * m_stride];
	}
	
	/// Element access.
	virtual const T &operator[](size_t i) const override
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
	
	/// Get the first element.
	T &front()
	{
		return m_buffer[0];
	}
	
	/// Get the last element.
	T &back()
	{
		return m_buffer[(m_size - 1) * m_stride];
	}
	
	/// Get the first element.
	const T &front() const
	{
		return m_buffer[0];
	}
	
	/// Get the last element.
	const T &back() const
	{
		return m_buffer[(m_size - 1) * m_stride];
	}
	
	/// Element-wise addition.
	void add(const Vector &v, const T &scalar)
	{
		NNAssert(m_size == v.m_size, "Incompatible addends!");
		BLAS<T>::axpy(m_size, scalar, v.m_buffer, v.m_stride, m_buffer, m_stride);
	}
	
	/// Element-wise addition.
	Vector &operator+=(const Vector &v)
	{
		NNAssert(m_size == v.m_size, "Incompatible addends!");
		add(v, 1);
		return *this;
	}
	
	/// Operation addition.
	Vector &operator+=(const Operation<Vector> &op)
	{
		op.add(*this);
		return *this;
	}
	
	/// Element-wise subtraction.
	Vector &operator-=(const Vector &v)
	{
		NNAssert(m_size == v.m_size, "Incompatible addends!");
		add(v, -1);
		return *this;
	}
	
	/// Operation subtraction.
	Vector &operator-=(const Operation<Vector> &op)
	{
		op.sub(*this);
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
public:
	using Tensor<T>::shareBuffer;
	
	/// Identity matrix.
	static Matrix identity(size_t rows, size_t cols)
	{
		Matrix m(rows, cols, 0);
		size_t d = std::min(rows, cols);
		for(size_t i = 0; i < d; ++i)
			m(i, i) = 1;
		return m;
	}
	
	/// General-purpose constructor.
	Matrix(size_t rows, size_t cols, const T &val = T())
	: Tensor<T>(rows * cols, val), m_rows(rows), m_cols(cols), m_ld(cols)
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
	
	/// Shared-data constructor.
	Matrix(Tensor<T> &t, size_t rows, size_t cols, size_t offset = 0)
	: Tensor<T>(t, rows * cols, offset), m_rows(rows), m_cols(cols), m_ld(cols)
	{}
	
	/// Operation constructor.
	Matrix(const Operation<Matrix> &op)
	: Tensor<T>(0), m_rows(0), m_cols(0), m_ld(0)
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
		Tensor<T>::resize(rows * cols);
		m_size = rows * cols;
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
	
	/// Share data with the given Matrix.
	void share(Matrix &m)
	{
		shareBuffer(m, m.m_size);
		resize(m.m_rows, m.m_cols);
	}
	
	/// Fill this tensor with the given value.
	void fill(const T &val)
	{
		T *buffer = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, buffer += m_ld)
			BLAS<T>::set(m_cols, val, buffer, 1);
	}
	
	/// Row access.
	Vector<T> row(size_t i)
	{
		return Vector<T>(*this, m_cols, m_ld * i);
	}
	
	/// Row access.
	const Vector<T> row(size_t i) const
	{
		return Vector<T>(*const_cast<Matrix *>(this), m_cols, m_ld * i);
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
	
	/// Length of the leading dimension (usually m_cols).
	size_t ld() const
	{
		return m_ld;
	}
	
	/// Element-wise addition.
	void add(const Matrix &m, const T &scalar = 1)
	{
		NNAssert(m_rows == m.m_rows && m_cols == m.m_cols, "Incompatible addends!");
		T *from = m.m_buffer, *to = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, from += m.m_ld, to += m_ld)
			BLAS<T>::axpy(m_cols, scalar, from, 1, to, 1);
	}
	
	/// Element-wise addition (repeat the vector for each row).
	void add(const Vector<T> &v, const T &scalar = 1)
	{
		NNAssert(m_cols == v.m_size && v.m_stride == 1, "Incompatible addends!");
		T *from = v.m_buffer, *to = m_buffer;
		for(size_t i = 0; i < m_rows; ++i, to += m_ld)
			BLAS<T>::axpy(m_cols, scalar, from, 1, to, 1);
	}
	
	/// Element-wise addition.
	Matrix &operator+=(const Matrix &m)
	{
		add(m, 1);
		return *this;
	}
	
	/// Element-wise addition (repeat the vector for each row).
	Matrix &operator+=(const Vector<T> &v)
	{
		add(v, 1);
		return *this;
	}
	
	/// Operation addition.
	Matrix &operator+=(const Operation<Matrix> &op)
	{
		op.add(*this);
		return *this;
	}
	
	/// Element-wise subtraction.
	Matrix &operator-=(const Matrix &m)
	{
		add(m, -1);
		return *this;
	}
	
	/// Element-wise subtraction (repeat the vector for each row).
	Matrix &operator-=(const Vector<T> &v)
	{
		add(v, -1);
		return *this;
	}
	
	/// Operation subtraction.
	Matrix &operator-=(const Operation<Matrix> &op)
	{
		op.sub(*this);
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
	
	/// Matrix multiplication.
	void multiply(const Matrix &lhs, const Matrix &rhs, const T &alpha = 1, const T &beta = 0)
	{
		resize(lhs.m_rows, rhs.m_cols);
		NNAssert(lhs.m_cols == rhs.m_rows, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, lhs.m_rows, rhs.m_cols, lhs.m_cols, alpha, lhs.m_buffer, lhs.m_ld, rhs.m_buffer, rhs.m_ld, beta, m_buffer, m_ld);
	}
	
	/// Matrix multiplication (with transposition on the LHS).
	void multiply(const OperationTrans<Matrix> &lhs, const Matrix &rhs, const T &alpha = 1, const T &beta = 0)
	{
		resize(lhs.m_target.m_cols, rhs.m_cols);
		NNAssert(lhs.m_target.m_rows == rhs.m_rows, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasTrans, CblasNoTrans, lhs.m_target.m_cols, rhs.m_cols, lhs.m_target.m_rows, alpha, lhs.m_target.m_buffer, lhs.m_target.m_ld, rhs.m_buffer, rhs.m_ld, beta, m_buffer, m_ld);
	}
	
	/// Matrix multiplication (with transposition on the RHS).
	void multiply(const Matrix &lhs, const OperationTrans<Matrix> &rhs, const T &alpha = 1, const T &beta = 0)
	{
		resize(lhs.m_rows, rhs.m_target.m_rows);
		NNAssert(lhs.m_cols == rhs.m_target.m_cols, "Incompatible multiplicands!");
		BLAS<T>::gemm(CblasRowMajor, CblasNoTrans, CblasTrans, lhs.m_rows, rhs.m_target.m_rows, lhs.m_cols, alpha, lhs.m_buffer, lhs.m_ld, rhs.m_target.m_buffer, rhs.m_target.m_ld, beta, m_buffer, m_ld);
	}
	
private:
	size_t m_rows, m_cols, m_ld;
};

}

#endif

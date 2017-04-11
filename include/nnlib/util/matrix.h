#ifndef MATRIX_H
#define MATRIX_H

#include "tensor.h"
#include "vector.h"
#include "algebra.h"
#include "random.h"

namespace nnlib
{

template <typename T = double>
class Matrix : public Tensor<T>
{
using Tensor<T>::m_ptr;
using Tensor<T>::m_size;
using Tensor<T>::m_shared;
public:
	// MARK: Iterator
	
	/// Element iterator
	class Iterator : public std::iterator<std::forward_iterator_tag, T>
	{
	public:
		Iterator(T *ptr, size_t cols, size_t ld)	: m_ptr(ptr), m_cols(cols), m_ld(ld), m_col(0) {}
		Iterator &operator++()						{ if(++m_col % m_cols == 0) { m_col = 0; m_ptr += m_ld; } return *this; }
		T &operator*()								{ return m_ptr[m_col]; }
		bool operator==(const Iterator &i)			{ return m_ptr == i.m_ptr && m_cols == i.m_cols && m_ld == i.m_ld && m_col == i.m_col; }
		bool operator!=(const Iterator &i)			{ return m_ptr != i.m_ptr || m_cols != i.m_cols || m_ld != i.m_ld || m_col != i.m_col; }
	private:
		T *m_ptr;
		size_t m_cols, m_ld, m_col;
	};
	
	/// Element iterator
	class ConstIterator : public std::iterator<std::forward_iterator_tag, T>
	{
	public:
		ConstIterator(const T *ptr, size_t cols, size_t ld)	: m_ptr(ptr), m_cols(cols), m_ld(ld), m_col(0) {}
		ConstIterator &operator++()							{ if(++m_col % m_cols == 0) { m_col = 0; m_ptr += m_ld; } return *this; }
		const T &operator*()								{ return m_ptr[m_col]; }
		bool operator==(const ConstIterator &i)				{ return m_ptr == i.m_ptr && m_cols == i.m_cols && m_ld == i.m_ld && m_col == i.m_col; }
		bool operator!=(const ConstIterator &i)				{ return m_ptr != i.m_ptr || m_cols != i.m_cols || m_ld != i.m_ld || m_col != i.m_col; }
	private:
		const T *m_ptr;
		size_t m_cols, m_ld, m_col;
	};
	
	// MARK: Algebra; static methods
	
	/// Matrix-matrix multiplication.
	static void multiply(const Matrix &A, const Matrix &B, Matrix &C, bool transposeA = false, bool transposeB = false, T alpha = 1, T beta = 0)
	{
		NNAssert(&A != &C && &B != &C, "Product holder cannot be an operand!");
		NNAssert(transposeA ? A.m_cols == C.m_rows : A.m_rows == C.m_rows, "Incorrect product rows!");
		NNAssert(transposeB ? B.m_rows == C.m_cols : B.m_cols == C.m_cols, "Incorrect product cols!");
		NNAssert(
			(transposeA && transposeB) ? A.m_rows == B.m_cols :
			(transposeA ? A.m_rows == B.m_rows :
			(transposeB ? A.m_cols == B.m_cols : A.m_cols == B.m_rows)),
			"Incompatible operands!"
		);
		Algebra<T>::instance().gemm(
			CblasRowMajor,
			transposeA ? CblasTrans : CblasNoTrans,
			transposeB ? CblasTrans : CblasNoTrans,
			transposeA ? A.m_cols : A.m_rows,
			transposeB ? B.m_rows : B.m_cols,
			transposeA ? A.m_rows : A.m_cols,
			alpha, A.m_ptr, A.m_ld, B.m_ptr, B.m_ld, beta, C.m_ptr, C.m_ld
		);
	}
	
	/// Matrix-vector multiplication.
	static void multiply(const Matrix &A, const Vector<T> &B, Vector<T> &C, bool transpose = false, T alpha = 1, T beta = 0)
	{
		NNAssert(transpose && A.m_rows == B.m_size || !transpose && A.m_cols == B.m_size, "Incompatible matrix-vector operands!");
		NNAssert(transpose && A.m_cols == C.m_size || !transpose && A.m_rows == C.m_size, "Incompatible output vector for gemv!");
		Algebra<T>::instance().gemv(
			CblasRowMajor,
			transpose ? CblasTrans : CblasNoTrans,
			A.m_rows,
			A.m_cols,
			alpha, A.m_ptr, A.m_ld, B.m_ptr, B.m_stride, beta, C.m_ptr, C.m_stride
		);
	}
	
	/// Add vector-vector outer product.
	static void addOuterProduct(const Vector<T> &A, const Vector<T> &B, Matrix &C, T alpha = 1)
	{
		Algebra<T>::instance().ger(
			CblasRowMajor,
			C.m_rows, C.m_cols,
			alpha, A.m_ptr, A.m_stride, B.m_ptr, B.m_stride, C.m_ptr, C.m_ld
		);
	}
	
	/// Shuffle the rows of two matrices together.
	static void shuffleRows(const Matrix<T> &A, const Matrix<T> &B)
	{
		NNAssert(A.m_rows == B.m_rows, "Cannot shuffle matrices with a different number of rows!");
		for(size_t i = A.m_rows - 1; i > 0; --i)
		{
			size_t j = Random<size_t>::uniform(i);
			Algebra<T>::instance().swap(A.m_cols, A.m_ptr + i * A.m_ld, 1, A.m_ptr + j * A.m_ld, 1);
			Algebra<T>::instance().swap(B.m_cols, B.m_ptr + i * B.m_ld, 1, B.m_ptr + j * B.m_ld, 1);
		}
	}
	
	/// Shuffle the rows of two matrices together, where the rows of one matrix must stay in blocks.
	static void shuffleRows(const Matrix<T> &A, const Matrix<T> &B, size_t blockSizeA)
	{
		NNAssert(A.m_rows == B.m_rows * blockSizeA, "Cannot shuffle matrices with an incompatible number of rows!");
		NNAssert(A.m_rows % blockSizeA == 0, "Invalid block size!");
		for(size_t i = B.m_rows - 1; i > 0; --i)
		{
			size_t j = Random<size_t>::uniform(i);
			for(size_t k = 0; k < blockSizeA; ++k)
				Algebra<T>::instance().swap(A.m_cols, A.m_ptr + (i * blockSizeA + k) * A.m_ld, 1, A.m_ptr + (j * blockSizeA + k) * A.m_ld, 1);
			Algebra<T>::instance().swap(B.m_cols, B.m_ptr + i * B.m_ld, 1, B.m_ptr + j * B.m_ld, 1);
		}
	}
	
	// MARK: Constructors
	
	/// Create a matrix of size rows * cols.
	Matrix(size_t rows = 0, size_t cols = 0, const T &val = T()) : Tensor<T>(rows * cols), m_rows(rows), m_cols(cols), m_ld(cols)
	{
		fill(val);
	}
	
	/// Create a shallow copy of another matrix.
	Matrix(const Matrix &m) : Tensor<T>(m), m_rows(m.m_rows), m_cols(m.m_cols), m_ld(m.m_ld) {}
	
	/// Create a shallow copy of another matrix.
	Matrix &operator=(const Matrix &m)
	{
		Tensor<T>::operator=(m);
		m_rows = m.m_rows;
		m_cols = m.m_cols;
		m_ld = m.m_ld;
		return *this;
	}
	
	/// Create a shallow copy of a non-matrix tensor.
	Matrix(const Tensor<T> &t, size_t rows, size_t cols) : Tensor<T>(t), m_rows(rows), m_cols(cols), m_ld(cols) {}
	
	/// Create a matrix as a row vector.
	/// \todo maybe column vector is more flexible, since m_ld = stride?
	Matrix(const Vector<T> &v) : Tensor<T>(v), m_rows(1), m_cols(v.size()), m_ld(v.size())
	{
		NNAssert(v.stride() == 1, "Cannot make a matrix from a non-contiguous vector!");
	}
	
	// MARK: Element Manipulation
	
	Matrix &fill(const T &val)
	{
		std::fill(begin(), end(), val);
		return *this;
	}
	
	Matrix &resize(size_t rows, size_t cols)
	{
		Tensor<T>::resize(rows * cols);
		m_rows = rows;
		m_cols = cols;
		m_ld = cols;
		return *this;
	}
	
	// MARK: Non-static Algebra
	
	Matrix &add(const Matrix &m, T scalar = 1)
	{
		NNAssert(m.m_rows == m_rows && m.m_cols == m_cols, "Incompatible matrices for element-wise addition!");
		if(m_cols == m_ld && m.m_cols == m.m_ld)
			Algebra<T>::instance().axpy(m_size, scalar, m.m_ptr, 1, m_ptr, 1);
		else
			for(size_t i = 0; i < m_rows; ++i)
				Algebra<T>::instance().axpy(m_cols, scalar, m.m_ptr + i * m.m_ld, 1, m_ptr + i * m_ld, 1);
		return *this;
	}
	
	Matrix &scale(T scalar)
	{
		if(m_cols == m_ld)
			Algebra<T>::instance().scal(m_size, scalar, m_ptr, 1);
		else
			for(size_t i = 0; i < m_rows; ++i)
				Algebra<T>::instance().scal(m_cols, scalar, m_ptr + i * m_ld, 1);
		return *this;
	}
	
	Matrix &normalizeColumns(T min = 0.0, T max = 1.0)
	{
		for(size_t i = 0; i < m_cols; ++i)
			column(i).normalize(min, max);
		return *this;
	}
	
	Matrix &normalizeRows(T min = 0.0, T max = 1.0)
	{
		for(size_t i = 0; i < m_rows; ++i)
			row(i).normalize(min, max);
		return *this;
	}
	
	// MARK: Statistics
	
	T sum()
	{
		T d = 0;
		for(auto val : *this)
			d += val;
		return d;
	}
	
	// MARK: Row, Column, and Block Access
	
	/// Get a vector looking at the ith row in the matrix.
	Vector<T> operator[](size_t i)
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return Vector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	ConstVector<T> operator[](size_t i) const
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return ConstVector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	Vector<T> operator()(size_t i)
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return Vector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	const Vector<T> operator()(size_t i) const
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return Vector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	Vector<T> row(size_t i)
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return Vector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	const Vector<T> row(size_t i) const
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return Vector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the jth column in the matrix.
	Vector<T> column(size_t j)
	{
		NNAssert(j < m_cols, "Invalid Matrix column index!");
		return Vector<T>(*this, j, m_rows, m_ld);
	}
	
	/// Get a vector looking at the jth column in the matrix.
	ConstVector<T> column(size_t j) const
	{
		NNAssert(j < m_cols, "Invalid Matrix column index!");
		return ConstVector<T>(*this, j, m_rows, m_ld);
	}
	
	Matrix block(size_t row, size_t col, size_t rows = (size_t) -1, size_t cols = (size_t) -1) const
	{
		NNAssert(row < m_rows && col < m_cols, "Invalid row/col for matrix block!");
		Matrix m(*this);
		m.m_rows = std::min(m_rows - row, rows);
		m.m_cols = std::min(m_cols - col, cols);
		m.m_ptr += row * m_ld + col;
		return m;
	}
	
	Matrix &block(Matrix &m, size_t row, size_t col = 0, size_t rows = 0, size_t cols = 0) const
	{
		m.m_shared	= m_shared;
		m.m_rows	= rows == 0 ? m.m_rows : std::min(rows, m_rows - row);
		m.m_cols	= cols == 0 ? m.m_cols : std::min(cols, m_cols - col);
		m.m_ld		= m_ld;
		m.m_ptr		= m_ptr + row * m_ld + col;
		if(row + m.m_rows > m_rows)
			m.m_rows = m_rows - row;
		return m;
	}
	
	// MARK: Element Access
	
	T &operator()(size_t i, size_t j)
	{
		NNAssert(i < m_rows && j < m_cols, "Invalid Matrix indices!");
		return m_ptr[i * m_ld + j];
	}
	
	const T &operator()(size_t i, size_t j) const
	{
		NNAssert(i < m_rows && j < m_cols, "Invalid Matrix indices!");
		return m_ptr[i * m_ld + j];
	}
	
	// MARK: Other methods
	
	size_t rows() const
	{
		return m_rows;
	}
	
	size_t cols() const
	{
		return m_cols;
	}
	
	// MARK: Iterators
	
	Iterator begin()
	{
		return Iterator(m_ptr, m_cols, m_ld);
	}
	
	ConstIterator begin() const
	{
		return ConstIterator(m_ptr, m_cols, m_ld);
	}
	
	Iterator end()
	{
		return Iterator(m_ptr + m_ld * m_rows, m_cols, m_ld);
	}
	
	ConstIterator end() const
	{
		return ConstIterator(m_ptr + m_ld * m_rows, m_cols, m_ld);
	}
private:
	size_t m_rows, m_cols;	///< the dimensions of this Matrix
	size_t m_ld;			///< the length of the leading dimension (i.e. row stride)
};

}

#endif

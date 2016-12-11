#ifndef MATRIX_H
#define MATRIX_H

#include "tensor.h"
#include "vector.h"

namespace nnlib
{

template <typename T>
class Matrix : public Tensor<T>
{
using Tensor<T>::m_ptr;
using Tensor<T>::m_size;
public:
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
	
	// MARK: Constructors
	
	/// Create a matrix of size rows * cols.
	Matrix(size_t rows = 0, size_t cols = 0) : Tensor<T>(rows * cols), m_rows(rows), m_cols(cols), m_ld(cols)
	{
		fill(T());
	}
	
	/// Create a shallow copy of another matrix.
	Matrix(const Matrix &m) : Tensor<T>(m), m_rows(m.rows), m_cols(m.cols), m_ld(m.ld) {}
	
	// MARK: Element Manipulation
	
	void fill(const T &val)
	{
		std::fill(begin(), end(), val);
	}
	
	// MARK: Element Access
	/*
	T &operator[](size_t i)
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	
	const T&operator[](size_t i) const
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	
	T &operator()(size_t i)
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	
	const T &operator()(size_t i) const
	{
		NNAssert(i < m_size, "Invalid Vector index!");
		return m_ptr[i * m_stride];
	}
	*/
	
	// MARK: Iterators
	
	Iterator begin()
	{
		return Iterator(m_ptr, m_cols, m_ld);
	}
	
	Iterator end()
	{
		return Iterator(m_ptr + m_ld * m_rows, m_cols, m_ld);
	}
private:
	size_t m_rows, m_cols;	///< the dimensions of this Matrix
	size_t m_ld;			///< the length of the leading dimension (i.e. row stride)
};

}

#endif

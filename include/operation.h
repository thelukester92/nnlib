#ifndef OP_H
#define OP_H

#include <type_traits>

namespace nnlib
{

template <typename T>
class Operation
{
public:
	typedef T type;
	virtual ~Operation() {}
	virtual void assign(T &dest) const = 0;
	virtual void add(T &dest) const = 0;
	virtual void sub(T &dest) const = 0;
};

template <typename T, typename U, typename V>
class BinaryOperation : public Operation<T>
{
public:
	BinaryOperation(const U &lhs, const V &rhs) : m_lhs(lhs), m_rhs(rhs) {}
protected:
	const U &m_lhs;
	const V &m_rhs;
};

template <typename T, typename U, typename V>
class OperationAdd : public BinaryOperation<T, U, V>
{
using BinaryOperation<T, U, V>::BinaryOperation;
using BinaryOperation<T, U, V>::m_lhs;
using BinaryOperation<T, U, V>::m_rhs;
public:
	virtual void assign(T &dest) const override
	{
		dest.copy(m_lhs);
		dest += m_rhs;
	}
	
	virtual void add(T &dest) const override
	{
		dest += m_lhs;
		dest += m_rhs;
	}
	
	virtual void sub(T &dest) const override
	{
		dest -= m_lhs;
		dest -= m_rhs;
	}
};

template <typename T, typename U, typename V>
class OperationSub : public BinaryOperation<T, U, V>
{
using BinaryOperation<T, U, V>::BinaryOperation;
using BinaryOperation<T, U, V>::m_lhs;
using BinaryOperation<T, U, V>::m_rhs;
public:
	virtual void assign(T &dest) const override
	{
		dest.copy(m_lhs);
		dest -= m_rhs;
	}
	
	virtual void add(T &dest) const override
	{
		dest += m_lhs;
		dest -= m_rhs;
	}
	
	virtual void sub(T &dest) const override
	{
		dest -= m_lhs;
		dest += m_rhs;
	}
};

template <typename T, typename U, typename V>
class OperationMult : public BinaryOperation<T, U, V>
{
using BinaryOperation<T, U, V>::BinaryOperation;
using BinaryOperation<T, U, V>::m_lhs;
using BinaryOperation<T, U, V>::m_rhs;
public:
	virtual void assign(T &dest) const override
	{
		dest.multiply(m_lhs, m_rhs, 1, 0);
	}
	
	virtual void add(T &dest) const override
	{
		dest.multiply(m_lhs, m_rhs, 1, 1);
	}
	
	virtual void sub(T &dest) const override
	{
		dest.multiply(m_lhs, m_rhs, -1, 1);
	}
};

template <typename T, typename U>
class UnaryOperation : public Operation<T>
{
public:
	UnaryOperation(const U &target) : m_target(target) {}
protected:
	const U &m_target;
};

template <typename T, typename U>
class OperationNeg : public UnaryOperation<T, U>
{
using UnaryOperation<T, U>::UnaryOperation;
using UnaryOperation<T, U>::m_target;
public:
	virtual void assign(T &dest) const override
	{
		dest.copy(m_target);
		dest *= -1;
	}
	
	virtual void add(T &dest) const override
	{
		dest -= m_target;
	}
	
	virtual void sub(T &dest) const override
	{
		dest += m_target;
	}
};

/// \todo determine a more efficient way to directly assign transpositions.
///       this is a low-priority issue because assigning transpositions is
///       not generally used; instead, one should use them as multiplicands.
template <typename T, typename U = T>
class OperationTrans : public UnaryOperation<T, U>
{
friend T;
using UnaryOperation<T, U>::UnaryOperation;
using UnaryOperation<T, U>::m_target;
public:
	virtual void assign(T &dest) const override
	{
		T I = T::identity(m_target.cols(), m_target.cols());
		OperationMult<T, T, OperationTrans<T, U>>(I, *this).assign(dest);
	}
	
	virtual void add(T &dest) const override
	{
		T I = T::identity(m_target.cols(), m_target.cols());
		OperationMult<T, T, OperationTrans<T, U>>(I, *this).add(dest);
	}
	
	virtual void sub(T &dest) const override
	{
		T I = T::identity(m_target.cols(), m_target.cols());
		OperationMult<T, T, OperationTrans<T, U>>(I, *this).sub(dest);
	}
};

template <typename T, typename U = T>
class OperationSumRows : public UnaryOperation<T, U>
{
using UnaryOperation<T, U>::UnaryOperation;
using UnaryOperation<T, U>::m_target;
public:
	virtual void assign(T &dest) const override
	{
		T row = m_target.row(0);
		size_t ld = m_target.ld(), n = m_target.rows();
		size_t offset = ld;
		dest.copy(row);
		for(size_t i = 1; i < n; ++i, offset += ld)
		{
			row.setOffset(offset);
			dest += row;
		}
	}
	
	virtual void add(T &dest) const override
	{
		T row = m_target.row(0);
		size_t ld = m_target.ld(), n = m_target.rows(), offset = 0;
		for(size_t i = 0; i < n; ++i, offset += ld)
		{
			row.setOffset(offset);
			dest += row;
		}
	}
	
	virtual void sub(T &dest) const override
	{
		T row = m_target.row(0);
		size_t ld = m_target.ld(), n = m_target.rows(), offset = 0;
		for(size_t i = 0; i < n; ++i, offset += ld)
		{
			row.setOffset(offset);
			dest -= row;
		}
	}
};

// MARK: Operation result metaprogramming helper.

template <typename T>
struct OperationResult
{
	typedef T type;
};

template <typename T>
struct OperationResult<Operation<T>>
{
	typedef typename OperationResult<T>::type type;
};

template <typename T, typename U, typename V>
struct OperationResult<OperationAdd<T, U, V>>
{
	typedef typename OperationResult<T>::type type;
};

template <typename T, typename U, typename V>
struct OperationResult<OperationSub<T, U, V>>
{
	typedef typename OperationResult<T>::type type;
};

template <typename T, typename U, typename V>
struct OperationResult<OperationMult<T, U, V>>
{
	typedef typename OperationResult<T>::type type;
};

template <typename T, typename U>
struct OperationResult<OperationNeg<T, U>>
{
	typedef typename OperationResult<T>::type type;
};

template <typename T, typename U>
struct OperationResult<OperationTrans<T, U>>
{
	typedef typename OperationResult<T>::type type;
};

// MARK: Operator overloads.

/// Addition operator overload.
template <typename U, typename V>
OperationAdd<typename OperationResult<U>::type, U, V> operator+(const U &lhs, const V &rhs)
{
	return OperationAdd<typename OperationResult<U>::type, U, V>(lhs, rhs);
}

/// Subtraction operator overload.
template <typename U, typename V>
OperationSub<typename OperationResult<U>::type, U, V> operator-(const U &lhs, const V &rhs)
{
	return OperationSub<typename OperationResult<U>::type, U, V>(lhs, rhs);
}

/// (Matrix) multiplication operator overload.
template <typename U, typename V>
OperationMult<typename OperationResult<U>::type, U, V> operator*(const U &lhs, const V &rhs)
{
	return OperationMult<typename OperationResult<U>::type, U, V>(lhs, rhs);
}

/// Unary negation operator overload.
template <typename U>
OperationNeg<typename OperationResult<U>::type, U> operator-(const U &target)
{
	return OperationNeg<typename OperationResult<U>::type, U>(target);
}

/// Unary transposition operator overload.
template <typename U>
OperationTrans<typename OperationResult<U>::type, U> operator~(const U &target)
{
	return OperationTrans<typename OperationResult<U>::type, U>(target);
}

}

#endif

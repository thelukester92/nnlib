#ifndef OP_H
#define OP_H

#include "tensor.h"

namespace nnlib
{

template <typename T>
class Operation
{
public:
	virtual ~Operation() {}
	virtual void assign(T &dest) const = 0;
};

template <typename T, typename U = T, typename V = T>
class BinaryOperation : public Operation<T>
{
public:
	BinaryOperation(const U &lhs, const V &rhs) : m_lhs(lhs), m_rhs(rhs) {}
protected:
	const U &m_lhs;
	const V &m_rhs;
};

template <typename T>
class OperationAdd : public BinaryOperation<T>
{
using BinaryOperation<T>::BinaryOperation;
using BinaryOperation<T>::m_lhs;
using BinaryOperation<T>::m_rhs;
public:
	virtual void assign(T &dest) const override
	{
		dest = m_lhs;
		dest += m_rhs;
	}
};

}

#endif

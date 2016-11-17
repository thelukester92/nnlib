#ifndef OP_H
#define OP_H

#include "tensor.h"

namespace nnlib
{

class Operation
{};

template <typename T, typename U, typename V>
class BinaryOperation : public Operation
{
public:
	BinaryOperation(const U &lhs, const V &rhs) : m_lhs(lhs), m_rhs(rhs) {}
	virtual void assign(T &dest) = 0;
private:
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
	virtual void assign(T &dest) override
	{
		dest = m_lhs;
		dest += m_rhs;
	}
};

}

#endif

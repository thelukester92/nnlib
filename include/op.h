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
	virtual void apply(T &dest) = 0;
private:
	const U &m_lhs;
	const V &m_rhs;
};

template <typename T, typename U, typename V>
class foo;

}

#endif

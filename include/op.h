#ifndef OP_H
#define OP_H

namespace nnlib
{

template <typename LHS, typename RHS>
struct BinaryOperator
{
	BinaryOperator(const LHS &_lhs, const RHS &_rhs) : lhs(_lhs), rhs(_rhs) {}
	const LHS &lhs;
	const RHS &rhs;
};

template <typename LHS, typename RHS>
struct OperatorAdd : public BinaryOperator<LHS, RHS>
{
using BinaryOperator<LHS, RHS>::BinaryOperator;
};

template <typename LHS, typename RHS>
struct OperatorMultiply : public BinaryOperator<LHS, RHS>
{
using BinaryOperator<LHS, RHS>::BinaryOperator;
};

}

#endif

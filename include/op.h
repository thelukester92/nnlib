#ifndef OP_H
#define OP_H

namespace nnlib
{

template <typename T, typename U>
struct BinOp
{
	BinOp(const T &_lhs, const U &_rhs) : lhs(_lhs), rhs(_rhs) {}
	const T &lhs;
	const U &rhs;
};

template <typename T, typename U>
struct OpAdd : public BinOp<T, U>
{
using BinOp<T, U>::BinOp;
};

template <typename T, typename U>
struct OpSub : public BinOp<T, U>
{
using BinOp<T, U>::BinOp;
};

template <typename T, typename U>
struct OpMult : public BinOp<T, U>
{
using BinOp<T, U>::BinOp;
};

}

#endif

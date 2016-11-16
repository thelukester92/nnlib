#ifndef OP_H
#define OP_H

namespace nnlib
{

struct Op
{};

template <typename T>
struct UnaryOp : public Op
{
	UnaryOp(const T &_target) : target(_target) {}
	const T &target;
};

template <typename T>
struct OpTrans : public UnaryOp<T>
{
using UnaryOp<T>::UnaryOp;
};

template <typename T>
struct OpCollapse : public UnaryOp<T>
{
	OpCollapse(const T &_target, bool _collapseCols = false) : UnaryOp<T>(_target), collapseCols(_collapseCols) {}
	bool collapseCols;
};

template <typename T, typename U>
struct BinaryOp : public Op
{
	BinaryOp(const T &_lhs, const U &_rhs) : lhs(_lhs), rhs(_rhs) {}
	const T &lhs;
	const U &rhs;
};

template <typename T, typename U>
struct OpAdd : public BinaryOp<T, U>
{
using BinaryOp<T, U>::BinaryOp;
};

template <typename T, typename U>
struct OpSub : public BinaryOp<T, U>
{
using BinaryOp<T, U>::BinaryOp;
};

template <typename T, typename U>
struct OpMult : public BinaryOp<T, U>
{
using BinaryOp<T, U>::BinaryOp;
};

}

#endif

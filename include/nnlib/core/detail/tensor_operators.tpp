#ifndef CORE_TENSOR_OPERATORS_TPP
#define CORE_TENSOR_OPERATORS_TPP

#include "../tensor.hpp"
#include <iomanip>
#include <limits>
#include <sstream>

#ifndef NN_MAX_NUM_DIMENSIONS
#define NN_MAX_NUM_DIMENSIONS 32ul
#endif

#ifndef NN_MAX_PRECISION
#define NN_MAX_PRECISION 6ul
#endif

template <typename T>
std::ostream &operator<<(std::ostream &out, const nnlib::Tensor<T> &t)
{
    out << std::left;

    if(t.dims() == 1)
    {
        out.precision(std::numeric_limits<double>::digits10);
        for(size_t i = 0; i < t.size(0); ++i)
        {
            out << t(i) << "\n";
        }
    }
    else if(t.dims() == 2)
    {
        nnlib::Storage<size_t> width(t.size(1), 1);
        for(size_t i = 0; i < t.size(0); ++i)
        {
            for(size_t j = 0; j < t.size(1); ++j)
            {
                std::stringstream ss;
                ss.precision(NN_MAX_PRECISION);
                ss << t(i, j);
                size_t w = ss.str().size();
                if(t(i, j) < 0)
                    ++w;
                if(ss.str().find('.') != std::string::npos)
                    ++w;
                if(w > NN_MAX_PRECISION)
                {
                    if(t(i, j) > -10 && t(i, j) < 10)
                        width[j] = NN_MAX_PRECISION + 2;
                    else
                        width[j] = NN_MAX_PRECISION + 6;
                }
                else
                    width[j] = std::max(width[j], w);
            }
        }

        out.precision(NN_MAX_PRECISION);
        for(size_t i = 0; i < t.size(0); ++i)
        {
            for(size_t j = 0; j < t.size(1); ++j)
                out << std::setw(width[j] + 1) << t(i, j);
            out << "\n";
        }
    }

    out << "[ Tensor of dimension " << t.size(0);
    for(size_t i = 1; i < t.dims(); ++i)
    {
        out << " x " << t.size(i);
    }
    out << " ]";

    return out;
}

template <typename T>
nnlib::Tensor<T> &operator+=(nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
    forEach([&](T x, T &y)
    {
        y += x;
    }, rhs, lhs);
    return lhs;
}

template <typename T>
nnlib::Tensor<T> operator+(const nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
    nnlib::Tensor<T> sum = lhs.copy();
    return sum += rhs;
}

template <typename T>
nnlib::Tensor<T> &operator-=(nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
    forEach([&](T x, T &y)
    {
        y -= x;
    }, rhs, lhs);
    return lhs;
}

template <typename T>
nnlib::Tensor<T> operator-(const nnlib::Tensor<T> &lhs, const nnlib::Tensor<T> &rhs)
{
    nnlib::Tensor<T> difference = lhs.copy();
    return difference -= rhs;
}

template <typename T>
nnlib::Tensor<T> &operator*=(nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
    return lhs.scale(rhs);
}

template <typename T>
nnlib::Tensor<T> operator*(const nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
    nnlib::Tensor<T> product = lhs.copy();
    return product *= rhs;
}

template <typename T>
nnlib::Tensor<T> operator*(typename nnlib::traits::Identity<T>::type lhs, const nnlib::Tensor<T> &rhs)
{
    nnlib::Tensor<T> product = rhs.copy();
    return product *= lhs;
}

template <typename T>
nnlib::Tensor<T> &operator/=(nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
    return lhs.scale(1.0 / rhs);
}

template <typename T>
nnlib::Tensor<T> operator/(const nnlib::Tensor<T> &lhs, typename nnlib::traits::Identity<T>::type rhs)
{
    nnlib::Tensor<T> quotient = lhs.copy();
    return quotient /= rhs;
}

#endif

#ifndef UTIL_PROGRESS_HPP
#define UTIL_PROGRESS_HPP

#include <math.h>
#include <iostream>
#include "timer.hpp"

namespace nnlib
{

class Progress
{
public:
    inline Progress(size_t total, std::ostream &out = std::cout, size_t length = 50);
    inline void reset();
    inline void display(size_t current);

private:
    size_t m_total, m_degreeTotal, m_length;
    std::ostream &m_out;
    Timer m_timer;
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
    #include "detail/progress.tpp"
#endif

#endif

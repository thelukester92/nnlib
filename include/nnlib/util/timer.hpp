#ifndef UTIL_TIMER_HPP
#define UTIL_TIMER_HPP

#include <chrono>
#include <sstream>

namespace nnlib
{

class Timer
{
public:
    using clock = std::chrono::high_resolution_clock;

    Timer(std::chrono::time_point<clock> start = clock::now());
    void reset();
    double elapsed(bool startOver = false);
    std::string ftime(double t = -1);

private:
    std::chrono::time_point<clock> m_start;
};

}

#if !defined NN_REAL_T && !defined NN_IMPL
    #include "detail/timer.tpp"
#endif

#endif

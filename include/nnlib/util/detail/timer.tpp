#ifndef UTIL_TIMER_TPP
#define UTIL_TIMER_TPP

#include "../timer.hpp"
#include <iomanip>

namespace nnlib
{

Timer::Timer(std::chrono::time_point<Timer::clock> start) :
    m_start(start)
{}

void Timer::reset()
{
    m_start = clock::now();
}

double Timer::elapsed(bool startOver)
{
    double span = std::chrono::duration<double>(clock::now() - m_start).count();
    if(startOver)
        reset();
    return span;
}

std::string Timer::ftime(double t)
{
    if(t < 0)
        t = elapsed();

    std::ostringstream out;
    out << std::setprecision(1) << std::fixed;

    size_t m = t / 60;
    t -= m * 60;

    size_t h = m / 60;
    m -= h * 60;

    size_t d = h / 24;
    h -= d * 24;

    if(d > 0)
        out << d << "d ";
    if(d > 0 || h > 0)
        out << h << "h ";
    if(d > 0 || h > 0 || m > 0)
        out << m << "m ";
    out << t << "s";

    return out.str();
}

}

#endif

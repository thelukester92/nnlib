#ifndef UTIL_TIMER_HPP
#define UTIL_TIMER_HPP

#include <chrono>
#include <iomanip>
#include <sstream>

namespace nnlib
{

class Timer
{
using clock = std::chrono::high_resolution_clock;
public:
    Timer(std::chrono::time_point<clock> start = clock::now())
        : m_start(start)
    {}

    void reset()
    {
        m_start = clock::now();
    }

    double elapsed(bool startOver = false)
    {
        double span = std::chrono::duration<double>(clock::now() - m_start).count();
        if(startOver)
            reset();
        return span;
    }

    std::string ftime(double t = -1)
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

private:
    std::chrono::time_point<clock> m_start;
};

}

#endif

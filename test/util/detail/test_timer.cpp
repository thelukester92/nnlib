#include "../test_timer.hpp"
#include "nnlib/util/timer.hpp"
using namespace nnlib;

NNTestClassImpl(Timer)
{
    NNTestMethod(Timer)
    {
        NNTestParams(std::chrono::time_point)
        {
            Timer t(Timer::clock::now() - std::chrono::seconds(30));
            NNTestAlmostEquals(t.elapsed(), 30, 0.1);
            Timer u;
            NNTestAlmostEquals(u.elapsed(), 0, 0.1);
        }
    }

    NNTestMethod(reset)
    {
        NNTestParams()
        {
            Timer t(Timer::clock::now() - std::chrono::seconds(30));
            t.reset();
            NNTestAlmostEquals(t.elapsed(), 0, 0.1);
        }
    }

    NNTestMethod(elapsed)
    {
        NNTestParams(bool)
        {
            Timer t(Timer::clock::now() - std::chrono::seconds(30));
            NNTestAlmostEquals(t.elapsed(true), 30, 0.1);
            NNTestAlmostEquals(t.elapsed(), 0, 0.1);
            Timer u(Timer::clock::now() - std::chrono::seconds(30));
            NNTestAlmostEquals(u.elapsed(false), 30, 0.1);
            NNTestAlmostEquals(u.elapsed(), 30, 0.1);
        }
    }

    NNTestMethod(ftime)
    {
        NNTestParams(double)
        {
            Timer t;
            NNTestEquals(t.ftime(), "0.0s");
            NNTestEquals(t.ftime(10.125), "10.1s");
            NNTestEquals(t.ftime(310.125), "5m 10.1s");
            NNTestEquals(t.ftime(7510.125), "2h 5m 10.1s");
            NNTestEquals(t.ftime(93910.125), "1d 2h 5m 10.1s");
        }
    }
}

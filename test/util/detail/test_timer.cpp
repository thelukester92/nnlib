#include "../test_timer.hpp"
#include "nnlib/util/timer.hpp"
using namespace nnlib;

NNTestClassImpl(Timer)
{
    NNTestMethod(Timer)
    {
        NNTestParams(std::chrono::time_point)
        {
            
        }
    }
}

#include "../test_random.hpp"
#include "nnlib/core/tensor.hpp"
#include "nnlib/math/math.hpp"
#include "nnlib/util/random.hpp"
using namespace nnlib;

void TestRandom()
{
    RandomEngine::sharedEngine().seed();
    double r;

    r = Random<NN_REAL_T>::sharedRandom().uniform(3.14);
    NNAssertGreaterThanOrEquals(r, 0, "Random::uniform(T) produced a value too small!");
    NNAssertLessThan(r, 3.14, "Random::uniform(T) produced a value too big!");

    r = Random<NN_REAL_T>::sharedRandom().uniform(2.19, 3.14);
    NNAssertGreaterThanOrEquals(r, 2.19, "Random::uniform(T, T) produced a value too small!");
    NNAssertLessThan(r, 3.14, "Random::uniform(T, T) produced a value too big!");

    Tensor<NN_REAL_T> t(1000);
    for(auto &v : t)
        v = Random<NN_REAL_T>::sharedRandom().normal();
    NNAssertAlmostEquals(math::mean(t), 0.0, 1e-1, "Random::normal produced an unexpected mean!");
    NNAssertAlmostEquals(math::variance(t), 1.0, 1e-1, "Random::normal produced an unexpected variance!");

    r = Random<NN_REAL_T>::sharedRandom().normal(0, 1, 1);
    NNAssertGreaterThanOrEquals(r, -1.0, "Random::normal(T, T, T) produced a value too small!");
    NNAssertLessThanOrEquals(r, 1.0, "Random::uniform(T, T, T) produced a value too big!");
}

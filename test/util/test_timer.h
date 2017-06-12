#ifndef TEST_TIMER_H
#define TEST_TIMER_H

#include "nnlib/util/timer.h"
using namespace nnlib;

void TestTimer()
{
	Timer t;
	t.reset();
	NNAssertLessThan(t.elapsed(), 1e-3, "Timer::elapsed returned a nonsensical value!");
}

#endif

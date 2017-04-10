#ifndef TIMER_H
#define TIMER_H

#include <chrono>

namespace nnlib
{

template <typename T = double>
class Timer
{
using clock = std::chrono::high_resolution_clock;
public:
	Timer(std::chrono::time_point<clock> start = clock::now()) : m_start(start) {}
	void reset()
	{
		m_start = clock::now();
	}
	T elapsed(bool startOver = false)
	{
		T span = std::chrono::duration<T>(clock::now() - m_start).count();
		if(startOver)
			reset();
		return span;
	}
private:
	std::chrono::time_point<clock> m_start;
};

}

#endif

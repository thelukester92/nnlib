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
	Progress(size_t total, size_t length = 50, std::ostream &out = std::cout) :
		m_total(total),
		m_degreeTotal((size_t) ceil(log(total) / log(10)) + 1),
		m_length(length),
		m_out(out)
	{}
	
	void reset()
	{
		m_timer.reset();
	}
	
	void display(size_t current)
	{
		size_t degreeCurrent	= current == 0 ? 1 : (size_t) ceil(log(current + 1) / log(10));
		size_t middle			= degreeCurrent + m_degreeTotal + 5;
		size_t head				= current == m_total ? m_length + 1 : m_length * current / double(m_total);
		size_t leading			= (m_length - middle) / 2;
		size_t trailing			= m_length - middle - leading;
		
		m_out << "\r\33[2K[";
		for(size_t i = 1; i < leading; ++i)
			m_out << (i < head ? "=" : (i == head ? ">" : " "));
		m_out << " " << current << " / " << m_total << " ";
		for(size_t i = leading + middle; i <= leading + middle + trailing; ++i)
			m_out << (i < head ? "=" : (i == head ? ">" : " "));
		m_out << "]";
		
		if(current < m_total)
		{
			m_out << " " << m_timer.ftime();
			if(current > 0)
				m_out << " / " << m_timer.ftime(m_timer.elapsed() / current * m_total);
		}
		else
			m_out << " Done in " << m_timer.ftime() << "! ^_^\n";
		
		m_out << std::flush;
	}
	
private:
	size_t m_total, m_degreeTotal, m_length;
	std::ostream &m_out;
	Timer m_timer;
};

}

#endif

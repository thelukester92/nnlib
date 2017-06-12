#ifndef PROGRESS_H
#define PROGRESS_H

#include <math.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "timer.h"

namespace nnlib
{

class Progress
{
public:
	Progress() = delete;
	
	static void display(size_t current, size_t total, char end = '\0', size_t length = 50, std::ostream &out = std::cout)
	{
		size_t degreeCurrent	= current == 0 ? 1 : (size_t) ceil(log(current + 1) / log(10));
		size_t degreeTotal		= (size_t) ceil(log(total) / log(10)) + 1;
		size_t middle			= degreeCurrent + degreeTotal + 5;
		size_t head				= current == total ? length+1 : length * current / double(total);
		size_t leading			= (length - middle) / 2;
		size_t trailing			= length - middle - leading;
		
		if(current == 0)
			timer.reset();
		
		out << "\r\33[2K[";
		for(size_t i = 1; i < leading; ++i)
			out << (i < head ? "=" : (i == head ? ">" : " "));
		out << " " << current << " / " << total << " ";
		for(size_t i = leading + middle; i <= leading + middle + trailing; ++i)
			out << (i < head ? "=" : (i == head ? ">" : " "));
		out << "]";
		
		if(current < total)
		{
			out << " " << ftime(timer.elapsed());
			if(current > 0)
				out << " / " << ftime(timer.elapsed() / current * total);
		}
		else
			out << " Done in " << ftime(timer.elapsed()) << "! ^_^";
		
		out << end << std::flush;
	}
	
private:
	static std::string ftime(double t)
	{
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
	
	static Timer timer;
};

Timer Progress::timer;

}

#endif

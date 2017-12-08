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
	Progress() = delete;
	
	static void display(size_t current, size_t total, std::ostream &out = std::cout, size_t length = 50)
	{
		static Timer timer;
		
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
			out << " " << timer.ftime();
			if(current > 0)
				out << " / " << timer.ftime(timer.elapsed() / current * total);
		}
		else
			out << " Done in " << timer.ftime() << "! ^_^\n";
		
		out << std::flush;
	}
};

}

#endif

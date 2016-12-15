#ifndef PROGRESS_H
#define PROGRESS_H

#include <math.h>
#include <iostream>

namespace nnlib
{

class Progress
{
public:
	static void display(size_t current, size_t total, char end = '\0', size_t length = 100, std::ostream &out = std::cout)
	{
		size_t degreeCurrent	= current == 0 ? 1 : (size_t) ceil(log(current + 1) / log(10));
		size_t degreeTotal		= (size_t) ceil(log(total) / log(10)) + 1;
		size_t middle			= degreeCurrent + degreeTotal + 5;
		size_t head				= current == total ? length+1 : length * (current + 1) / double(total);
		size_t leading			= (length - middle) / 2;
		size_t trailing			= length - middle - leading;
		
		out << "\r[";
		for(size_t i = 1; i < leading; ++i)
			out << (i < head ? "=" : (i == head ? ">" : " "));
		out << " " << current << " / " << total << " ";
		for(size_t i = leading + middle; i <= leading + middle + trailing; ++i)
			out << (i < head ? "=" : (i == head ? ">" : " "));
		out << "]" << end << std::flush;
	}
};

}

#endif
